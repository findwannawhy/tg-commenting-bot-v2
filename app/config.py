from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=".env",
		case_sensitive=True,
		protected_namespaces=("settings_",),
		extra="ignore",
	)
	# Server
	admin_token: str = Field(..., alias="ADMIN_TOKEN")
	host: str = Field("0.0.0.0", alias="HOST")
	port: int = Field(8000, alias="PORT")
	log_buffer_size: int = Field(5000, alias="LOG_BUFFER_SIZE")

	# Session storage
	sessions_dir: str = Field("sessions", alias="SESSIONS_DIR")

	# OpenAI model default (UI value stored in DB; env provides only default)
	model_default: str = Field("gpt-5", alias="MODEL_DEFAULT")

	# Database
	database_url: str = Field("sqlite+aiosqlite:///./data.db", alias="DATABASE_URL")



settings = Settings()  # load on import for simplicity
