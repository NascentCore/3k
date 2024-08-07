# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
import asyncio
import os

from graphrag.index import run_pipeline, run_pipeline_with_config
from graphrag.index.config import PipelineCSVInputConfig, PipelineWorkflowReference
from graphrag.index.input import load_input

sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../_sample_data/"
)

shared_dataset = asyncio.run(
    load_input(
        PipelineCSVInputConfig(
            file_pattern=".*\\.csv$",
            base_dir=sample_data_dir,
            source_column="author",
            text_column="message",
            timestamp_column="date(yyyyMMddHHmmss)",
            timestamp_format="%Y%m%d%H%M%S",
            title_column="message",
        ),
    )
)


async def run_with_config():
    """Run a pipeline with a config file"""
    # We're cheap, and this is an example, lets just do 10
    dataset = shared_dataset.head(10)

    # load pipeline.yml in this directory
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # Grab the last result from the pipeline, should be our entity extraction
    tables = []
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    # Print the entities.  This will be a row for each text unit, each with a list of entities,
    # This should look pretty close to the python version, but since we're using an LLM
    # it will be a little different depending on how it feels about the text
    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


async def run_python():
    if (
        "EXAMPLE_OPENAI_API_KEY" not in os.environ
        and "OPENAI_API_KEY" not in os.environ
    ):
        msg = "Please set EXAMPLE_OPENAI_API_KEY or OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

    # We're cheap, and this is an example, lets just do 10
    dataset = shared_dataset.head(10)

    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            name="entity_extraction",
            config={
                "entity_extract": {
                    "strategy": {
                        "type": "graph_intelligence",
                        "llm": {
                            "type": "openai_chat",
                            "api_key": os.environ.get(
                                "EXAMPLE_OPENAI_API_KEY",
                                os.environ.get("OPENAI_API_KEY", None),
                            ),
                            "model": os.environ.get(
                                "EXAMPLE_OPENAI_MODEL", "gpt-3.5-turbo"
                            ),
                            "max_tokens": os.environ.get(
                                "EXAMPLE_OPENAI_MAX_TOKENS", 2500
                            ),
                            "temperature": os.environ.get(
                                "EXAMPLE_OPENAI_TEMPERATURE", 0
                            ),
                        },
                    }
                }
            },
        )
    ]

    # Grab the last result from the pipeline, should be our entity extraction
    tables = []
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)
    pipeline_result = tables[-1]

    # Print the entities.  This will be a row for each text unit, each with a list of entities
    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(run_python())
    asyncio.run(run_with_config())
