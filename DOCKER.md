# Docker Usage Instructions

## Building the Image

From the repository root:

```bash
docker build -t synthla-edu-v2 .
```

## Running Experiments

### Option 1: Mount your data directory

```bash
# Full experimental matrix (both datasets, all synthesizers)
docker run -v /path/to/your/data/raw:/app/data/raw \
           -v /path/to/output:/app/runs \
           synthla-edu-v2 \
           python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs

# Quick validation run
docker run -v /path/to/your/data/raw:/app/data/raw \
           -v /path/to/output:/app/runs \
           synthla-edu-v2 \
           python synthla_edu_v2.py --run-all --quick --raw-dir data/raw --out-dir runs
```

### Option 2: Copy data into container

```bash
# Build with data included (larger image)
docker build -t synthla-edu-v2-with-data .

# Run without volume mounts
docker run synthla-edu-v2-with-data \
           python synthla_edu_v2.py --run-all --quick --raw-dir data/raw --out-dir runs

# Copy results out
docker cp <container_id>:/app/runs ./runs
```

## Interactive Mode

```bash
# Open a shell inside the container
docker run -it -v /path/to/data/raw:/app/data/raw \
               -v /path/to/output:/app/runs \
               synthla-edu-v2 /bin/bash

# Then run commands manually
python synthla_edu_v2.py --dataset oulad --raw-dir data/raw --out-dir runs --quick
```

## Notes

- **Data not included**: You must mount or copy your OULAD/ASSISTments datasets
- **GPU support**: Add `--gpus all` to docker run for CUDA acceleration (requires nvidia-docker)
- **Memory**: Ensure Docker has at least 8GB RAM allocated
- **Output persistence**: Always mount `-v` the `runs/` directory to save results

## Alternative: Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  synthla-edu-v2:
    build: .
    volumes:
      - ./data/raw:/app/data/raw
      - ./runs:/app/runs
    command: python synthla_edu_v2.py --run-all --quick --raw-dir data/raw --out-dir runs
```

Then run:
```bash
docker-compose up
```
