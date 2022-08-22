import os

# Run API in Debug mode
API_DEBUG = True

# We store user submissions on this file
UPLOADS_FILEPATH = "uploads/new_applications.txt"
os.makedirs(os.path.basename(UPLOADS_FILEPATH), exist_ok=True)

# REDIS settings
# Queue name
REDIS_QUEUE = "job"
# Port
REDIS_PORT = 6379
# DB Id
REDIS_DB_ID = 0
# Host IP
REDIS_IP = "redis"
# Sleep parameters which manages the
# interval between requests to our redis queue
API_SLEEP = 0.05
