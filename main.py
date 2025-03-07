from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from transformers import pipeline
import re, json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GENERATION_ARGS = {
    "max_new_tokens": 256,
    "return_full_text": False,
    "temperature": 0.1,
    "do_sample": False,
    "top_k": 40,
    "top_p": 1.0,
}


class Phi4Model:
    def __init__(self):
        self.pipe = None

    def load(self):
        try:
            """Load the Phi-4 model."""
            self.pipe = pipeline(
                "text-generation",
                model="microsoft/phi-4",
                trust_remote_code=True,
                model_kwargs={"torch_dtype": "auto"},
                device_map="auto",
            )
            logger.info("Phi-4 model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Phi-4 model: {str(e)}")
            raise RuntimeError("Failed to load the Phi-4 model.") from e

    def generate(self, prompt: str, **kwargs):

        if self.pipe is None:
            logger.error("Model is not loaded. Cannot generate text.")
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        try:
            """Generate text dynamically using provided arguments."""
            args = {
                **GENERATION_ARGS,
                **{k: v for k, v in kwargs.items() if v is not None},
            }
            # Ensure `return_full_text` is included
            args.setdefault("return_full_text", GENERATION_ARGS["return_full_text"])

            return self.pipe(prompt, **args)[0]["generated_text"]

        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Text generation failed.")


def postprocessing(text):
    """Extracts the classification value from JSON inside a text block."""
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            json_str = match.group(1)  # Extract JSON content
            parsed_json = json.loads(json_str)  # Parse JSON
            return parsed_json  # Return parsed JSON
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in generated text.")
            return "Invalid JSON"
    logger.error("No JSON found in generated text.")
    return "No JSON Found"


class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(
        default=GENERATION_ARGS["max_new_tokens"],
        gt=0,
        le=512,
        description="Max new tokens",
    )
    temperature: float = Field(
        default=GENERATION_ARGS["temperature"],
        ge=0.0,
        le=1.0,
        description="Sampling temperature",
    )
    do_sample: bool = Field(
        default=GENERATION_ARGS["do_sample"], description="Enable sampling"
    )
    top_k: int = Field(
        default=GENERATION_ARGS["top_k"], ge=1, le=100, description="Top-k sampling"
    )
    top_p: float = Field(
        default=GENERATION_ARGS["top_p"],
        ge=0.0,
        le=1.0,
        description="Top-p sampling (nucleus)",
    )


# Create the FastAPI app
app = FastAPI()
# Create the Phi-4 model
model = Phi4Model()
# Load the Phi-4 model
try:
    model.load()
except RuntimeError as e:
    logger.critical(f"Application startup failed: {str(e)}")


@app.get("/healthcheck", status_code=200)
def healthcheck():
    """Health check endpoint to verify the service status."""
    return {"status": "ok"}


@app.post("/generate")
def generate(request: InferRequest):
    """Generate text based on input prompt and parameters."""
    try:
        generated_text = model.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        extracted_json = postprocessing(generated_text)

        # Build response
        response = {
            "data": [
                {
                    "prompt": request.prompt,
                    "output": [extracted_json],  # Output as a list
                    "params": {
                        "temperature": request.temperature,
                        "top_k": request.top_k,
                        "top_p": request.top_p,
                    }
                }
            ],
            "message": "ok"
        }

        return response

    except HTTPException as http_err:
        raise http_err  # Forward predefined HTTP errors
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")
