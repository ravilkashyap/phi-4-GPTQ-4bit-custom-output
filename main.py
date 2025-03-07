from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import re
import json
import logging
from vllm import LLM, SamplingParams
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default Generation Parameters
GENERATION_ARGS = {
    "max_tokens": 512,  # Updated as per schema
    "temperature": 0.1,
    "top_p": 1.0,
    "top_k": 40,
}


class InferlessPythonModel:
    def __init__(self):
        self.llm = None

    def initialize(self):
        try:
            self.llm = LLM(model="kaitchup/Phi-4-AutoRound-GPTQ-4bit", quantization="gptq")
            logger.info("Inferless model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize the Inferless model: {str(e)}")
            raise RuntimeError("Model initialization failed.") from e

    def infer(self, prompt: str, **kwargs):
        if self.llm is None:
            logger.error("Model is not initialized. Cannot generate text.")
            raise HTTPException(status_code=500, detail="Model is not initialized.")

        try:
            system_prompt = kwargs.get("system_prompt", "You are a friendly bot.")
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", GENERATION_ARGS["temperature"]),
                top_p=kwargs.get("top_p", GENERATION_ARGS["top_p"]),
                top_k=int(kwargs.get("top_k", GENERATION_ARGS["top_k"])),
                max_tokens=kwargs.get("max_tokens", GENERATION_ARGS["max_tokens"]),
            )

            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            outputs = self.llm.chat(conversation, sampling_params)
            result_output = [output.outputs[0].text for output in outputs]

            return {"generated_text": result_output[0]}
        
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Text generation failed.")

    def finalize(self):
        self.llm = None



def postprocessing(text):
    """Extracts the classification value from JSON inside a text block."""
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            json_str = match.group(1)  # Extract JSON content
            parsed_json = json.loads(json_str)  # Parse JSON
            return parsed_json  # Return parsed JSON
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in generated text.")
            return "Invalid JSON"
    logger.error("No JSON found in generated text.")
    logger.error(text)
    return "No JSON Found"


class InferRequest(BaseModel):
    prompt: List[str] = Field(
        default=[""],
        description="Input text prompt"
    )
    system_prompt: Optional[List[str]] = Field(
        default=[""],
        description="System prompt defining behavior"
    )
    temperature: Optional[List[float]] = Field(
        default=[GENERATION_ARGS["temperature"]],
        ge=0.0,
        le=1.0,
        description="Sampling temperature",
        example=[0.1]
    )
    top_p: Optional[List[float]] = Field(
        default=[GENERATION_ARGS["top_p"]],
        ge=0.0,
        le=1.0,
        description="Top-p sampling ",
        example=[0.1]
    )
    max_tokens: Optional[List[int]] = Field(
        default=[GENERATION_ARGS["max_tokens"]],
        gt=0,
        le=512,
        description="Max tokens to generate",
        example=[128]
    )
    top_k: Optional[List[int]] = Field(
        default=[GENERATION_ARGS["top_k"]],
        ge=1,
        le=100,
        description="Top-k sampling",
        example=[40]
    )


# Create the FastAPI app
app = FastAPI()
# Create and initialize the Inferless model
model = InferlessPythonModel()
try:
    model.initialize()
except RuntimeError as e:
    logger.critical(f"Application startup failed: {str(e)}")



@app.get("/healthcheck", status_code=200)
async def healthcheck():
    """Health check endpoint to verify the service status."""
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: InferRequest):
    """Generate text based on input prompt and parameters."""
    try:
        generated_text = model.infer(
            request.prompt[0],
            system_prompt=request.system_prompt[0] if request.system_prompt else None,
            temperature=request.temperature[0],
            top_p=request.top_p[0],
            max_tokens=request.max_tokens[0],
            top_k=request.top_k[0],
        )

        extracted_json = postprocessing(generated_text["generated_text"])
        
        messages_prompt = {}
        if request.system_prompt:
            messages_prompt["system"] = request.system_prompt[0]
        if request.prompt:
            messages_prompt["user"] = request.prompt[0]

        # Build response
        response = {
            "data": [
                {
                    "prompt": messages_prompt,
                    "output": [extracted_json],  # Output as a list
                    "params": {
                        "temperature": request.temperature[0],
                        "top_k": request.top_k[0],
                        "top_p": request.top_p[0],
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