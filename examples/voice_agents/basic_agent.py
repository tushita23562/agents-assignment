import logging
import asyncio

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()

# ---------------- CONFIG ---------------- #

IGNORE_WORDS = {
    "yeah",
    "yes",
    "ok",
    "okay",
    "hmm",
    "uh-huh",
    "right",
    "aha",
}

INTERRUPT_WORDS = {
    "stop",
    "wait",
    "no",
    "pause",
    "hold",
}

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact with users via voice. "
                "Keep responses concise and clear. "
                "Do not use emojis, markdown, or special characters. "
                "You are friendly, curious, and speak English."
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "Sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # ---------- STATE FLAGS ---------- #
    agent_is_speaking = False
    pending_interrupt = False
    # -------------------------------- #

    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=False,  # REQUIRED by assignment
        false_interruption_timeout=1.0,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # -------- AGENT AUDIO STATE -------- #

    @session.on("agent_audio_started")
    def _on_agent_audio_started():
        nonlocal agent_is_speaking
        agent_is_speaking = True

    @session.on("agent_audio_finished")
    def _on_agent_audio_finished():
        nonlocal agent_is_speaking
        agent_is_speaking = False

    # -------- USER SPEECH EVENTS ------- #

    @session.on("user_started_speaking")
    def _on_user_started_speaking():
        nonlocal pending_interrupt
        if agent_is_speaking:
            pending_interrupt = True  # VAD fired, wait for STT

    @session.on("user_transcript")
    def on_user_transcript(msg):
        asyncio.create_task(handle_transcript(msg))

    async def handle_transcript(msg):
        response = await llm.chat(msg.text)
        await session.say(response)

        normalized = text.lower().strip()
        words = normalized.split()

        contains_interrupt = any(w in INTERRUPT_WORDS for w in words)
        contains_only_ignore = all(w in IGNORE_WORDS for w in words)

        if agent_is_speaking:
            if contains_interrupt:
                pending_interrupt = False
                await session.interrupt()
                return

            if contains_only_ignore:
                pending_interrupt = False
                return

            # Any real sentence interrupts
            pending_interrupt = False
            await session.interrupt()
            return

        # Agent is silent → normal behavior
        pending_interrupt = False

    # ---------------------------------- #

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions()
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
