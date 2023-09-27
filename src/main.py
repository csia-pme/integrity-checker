import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from pydantic import Field
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag

# Imports required by the service's model
import pandas as pd
import io
import tomli
from datetime import datetime

settings = get_settings()


class MyService(Service):
    """
    Check for integrity of a dataset
    """

    # Any additional fields must be excluded for Pydantic to work
    model: object = Field(exclude=True)

    def __init__(self):
        super().__init__(
            name="Integrity Checker",
            slug="integrity-checker",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="dataset", type=[FieldDescriptionType.TEXT_CSV, FieldDescriptionType.TEXT_PLAIN]),
                FieldDescription(name="config", type=[FieldDescriptionType.TEXT_PLAIN]),

            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.TEXT_CSV]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.DATA_PREPROCESSING,
                    acronym=ExecutionUnitTagAcronym.DATA_PREPROCESSING,
                ),
            ]
        )

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields

        raw = str(data["dataset"].data)[2:-1]
        raw = raw.replace('\\t', ',').replace('\\n', '\n').replace('\\r', '\n')

        df = pd.read_csv(io.StringIO(raw))

        config_raw = str(data["config"].data)[2:-1]
        config_raw = config_raw.replace('\\t', ',').replace('\\n', '\n').replace('\\r', '\n')
        config = tomli.loads(config_raw)
        issues_from_config = self.apply_tests_from_config(df, config)

        csv_string = issues_from_config.to_csv(index=False)
        csv_bytes = csv_string.encode('utf-8')

        buf = io.BytesIO()
        buf.write(csv_bytes)

        return {
            "result": TaskData(
                data=buf.getvalue(),
                type=FieldDescriptionType.TEXT_CSV
            )
        }

    def apply_tests_from_config(self, df, config):
        # Initialize dataframe to store issues
        issues_df = df.copy()
        issues_df['Reason'] = ""
        issues_df['Problematic_Columns'] = ""

        # Apply duplicate detection
        if 'duplicates' in config['detection']:
            duplicate_cols = config['detection']['duplicates']['columns']
            mask = df.duplicated(subset=duplicate_cols, keep='first')
            issues_df.loc[mask, 'Reason'] += "Duplicate, "
            issues_df.loc[mask, 'Problematic_Columns'] += ", ".join(duplicate_cols) + "; "

        # Apply missing value detection
        if 'missing_values' in config['detection']:
            missing_value_cols = config['detection']['missing_values']['columns']
            mask = df[missing_value_cols].isnull().any(axis=1)
            issues_df.loc[mask, 'Reason'] += "Missing Value, "
            for col in missing_value_cols:
                col_mask = df[col].isnull()
                issues_df.loc[col_mask, 'Problematic_Columns'] += col + "; "

        # Apply space string detection
        if 'space_strings' in config['detection']:
            space_string_cols = config['detection']['space_strings']['columns']
            mask = df[space_string_cols].applymap(lambda x: isinstance(x, str) and '  ' in x).any(axis=1)
            issues_df.loc[mask, 'Reason'] += "Space String, "
            for col in space_string_cols:
                col_mask = df[col].str.contains('  ', na=False)
                issues_df.loc[col_mask, 'Problematic_Columns'] += col + "; "

        # Apply range error detection
        if 'range_errors' in config['detection']:
            for col in config['detection']['range_errors']['columns']:
                min_val = config['detection']['range_errors'][f'{col}_min']
                max_val = config['detection']['range_errors'][f'{col}_max']
                mask = ~df[col].between(min_val, max_val)
                issues_df.loc[mask, 'Reason'] += "Range Error, "
                issues_df.loc[mask, 'Problematic_Columns'] += col + "; "

        # Apply date inconsistency detection
        if 'date_inconsistency' in config['detection']:
            for col in config['detection']['date_inconsistency']['columns']:
                desired_format = config['detection']['date_inconsistency'][f'{col}_format']\
                    .replace("YYYY", "%Y").replace("MM", "%m").replace("DD", "%d")
                mask = ~df[col].apply(lambda x: self.is_valid_date(x, desired_format))
                issues_df.loc[mask, 'Reason'] += "Date Inconsistency, "
                issues_df.loc[mask, 'Problematic_Columns'] += col + "; "

        # Clean up the Reason and Problematic_Columns columns
        issues_df['Reason'] = issues_df['Reason'].str.rstrip(', ')
        issues_df['Problematic_Columns'] = issues_df['Problematic_Columns'].str.rstrip('; ')

        # Filter out rows without issues
        issues_df = issues_df[issues_df['Reason'] != ""]

        return issues_df

    def is_valid_date(self, date_str, format):
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False


api_description = """
This service checks the integrity of a dataset based on a configuration file, it can detect :
- duplicates
- missing values
- useless strings
- range errors
- date inconsistencies
"""
api_summary = """
Integrity Checker of a dataset
"""

# Define the FastAPI application with information
app = FastAPI(
    title="Integrity Checker",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "CSIA-PME",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=['Service'])
app.include_router(tasks_router, tags=['Tasks'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)


service_service: ServiceService | None = None


@app.on_event("startup")
async def startup_event():
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(f"Aborting service announcement after "
                                       f"{settings.engine_announce_retries} retries")

    # Announce the service to its engine
    asyncio.ensure_future(announce())


@app.on_event("shutdown")
async def shutdown_event():
    # Global variable
    global service_service
    my_service = MyService()
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)
