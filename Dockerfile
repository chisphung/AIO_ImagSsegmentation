FROM phucnguyen04/uit_car_racing_2024:v1

ARG USERNAME=appuser
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	# DISPLAY will usually be injected at runtime (-e DISPLAY=$DISPLAY)
	APP_HOME=/workspace/CarRacing

WORKDIR ${APP_HOME}

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source
COPY . .

RUN if ! id -u ${USER_ID} >/dev/null 2>&1; then \
	  groupadd -g ${GROUP_ID} ${USERNAME} && \
	  useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}; \
	fi && \
	chown -R ${USER_ID}:${GROUP_ID} ${APP_HOME}

USER ${USER_ID}

EXPOSE 11000

CMD ["python3", "main.py"]

# ===== Usage =====
# Build:
#   docker build -t groupkin .
# Run (with GPU, host net, X11 forward):
# docker run --name it-car \
#   -it \
#   --gpus all \
#   --network host \
#   -e DISPLAY=$DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
#   -p 11000:11000 \
#   groupkin:latest



