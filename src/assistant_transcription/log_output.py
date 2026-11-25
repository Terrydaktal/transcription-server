import copy
import logging
import re


service_logger = logging.getLogger("assistant.whisper")

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[36m"
ANSI_LIGHT_TEAL = "\033[1;96m"
ANSI_MAGENTA = "\033[1;35m"
ANSI_GREEN = "\033[1;32m"
ANSI_YELLOW = "\033[1;93m"
ANSI_BREAKDOWN_VALUE = "\033[93m"
ANSI_RED = "\033[1;31m"


class FriendlyLogSource(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.name == "assistant.whisper"
            and record.getMessage().startswith(("Delivery ", "Recording prepared:"))
        ):
            record.source = "Client"
        elif record.name in {"assistant.whisper", "uvicorn", "uvicorn.access", "uvicorn.error"}:
            record.source = "Server"
        else:
            record.source = "Application"
        return True


def ansi(text: str, colour: str) -> str:
    return f"{colour}{text}{ANSI_RESET}" if text else ""


def colour_timing_fields(output: str) -> str:
    coloured_fields = []
    in_breakdown = False
    for part in re.split(r"(\s+)", output):
        if not part or part.isspace():
            coloured_fields.append(part)
            continue

        leading = ""
        while part.startswith(("(", "[")):
            leading += part[0]
            part = part[1:]
            in_breakdown = True

        trailing = ""
        while part.endswith((")", "]")):
            trailing = part[-1] + trailing
            part = part[:-1]

        if part == "+":
            coloured_fields.append(f"{ansi(leading, ANSI_CYAN)}{ansi(part, ANSI_CYAN)}")
        elif "=" not in part:
            coloured_fields.append(f"{ansi(leading, ANSI_CYAN)}{ansi(part, ANSI_CYAN)}")
        else:
            key, value = part.split("=", 1)
            if key == "outcome":
                value_colour = ANSI_GREEN if value == "success" else ANSI_RED
            else:
                value_colour = ANSI_BREAKDOWN_VALUE if in_breakdown else ANSI_YELLOW
            key_colour = ANSI_CYAN if in_breakdown else ANSI_LIGHT_TEAL
            coloured_fields.append(
                f"{ansi(leading, ANSI_CYAN)}"
                f"{ansi(key, key_colour)}={ansi(value, value_colour)}"
            )

        if trailing:
            coloured_fields.append(ansi(trailing, ANSI_CYAN))
            in_breakdown = False
    return "".join(coloured_fields)


def colour_transcription_event(output: str) -> str:
    transcription_output, separator, timing_output = output.rpartition(" | ")
    if not separator or not timing_output.startswith("outcome="):
        transcription_output = output
        timing_output = ""

    match = re.match(
        r"^(Transcribed Raw) \(([^)]*)\): (.*)$",
        transcription_output,
        flags=re.DOTALL,
    )
    if not match:
        return output

    coloured = (
        f"{ansi(match.group(1), ANSI_CYAN)} "
        f"({ansi(match.group(2), ANSI_MAGENTA)}): "
        f"{ansi(match.group(3), ANSI_GREEN)}"
    )
    if timing_output:
        coloured += f" {ansi('|', ANSI_CYAN)} {colour_timing_fields(timing_output)}"
    return coloured


def service_log(*args, **kwargs) -> None:
    output = kwargs.get("sep", " ").join(str(arg) for arg in args)
    lowered = output.lower()
    if output.startswith(("ERROR", "Failed")):
        level = logging.ERROR
        color_message = ansi(output, ANSI_RED)
    elif (
        output.startswith(("WARNING", "CUDA OOM"))
        or output.startswith("Delivery refused:")
        or "refusing to type" in lowered
    ):
        level = logging.WARNING
        color_message = ansi(output, ANSI_YELLOW)
    else:
        level = logging.INFO
        if output.startswith("Transcribed Raw"):
            color_message = colour_transcription_event(output)
        elif output.startswith("Transcription timing"):
            prefix, _, timing = output.partition(" ")
            color_message = f"{ansi(prefix, ANSI_CYAN)} {colour_timing_fields(timing)}"
        elif output.startswith(
            ("Delivery complete: ", "Delivery refused: ", "Recording prepared: ")
        ):
            label, timing = output.split(": ", 1)
            color_message = (
                f"{ansi(label + ':', ANSI_CYAN)} {colour_timing_fields(timing)}"
            )
        elif output.startswith("Audio received: "):
            label, timing = output.split(": ", 1)
            color_message = (
                f"{ansi(label + ':', ANSI_GREEN)} {colour_timing_fields(timing)}"
            )
        else:
            color_message = output

    service_logger.log(level, output, extra={"color_message": color_message})


def build_log_config() -> dict:
    from uvicorn.config import LOGGING_CONFIG

    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = (
        "%(asctime)s.%(msecs)03d %(source)s: %(message)s"
    )
    log_config["formatters"]["access"]["fmt"] = (
        '%(asctime)s.%(msecs)03d %(source)s: %(client_addr)s - '
        '"%(request_line)s" %(status_code)s'
    )
    log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config.setdefault("filters", {})["friendly_source"] = {"()": FriendlyLogSource}
    for handler_name in ("default", "access"):
        log_config["handlers"][handler_name]["filters"] = ["friendly_source"]
    log_config["loggers"]["assistant.whisper"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    return log_config
