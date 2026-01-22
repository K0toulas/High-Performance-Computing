# Compiler and flags
NVCC = nvcc
CFLAGS = -O4 -g
DIRS =	Final
TARGET = $(DIRS)/clahe
SOURCES = $(DIRS)/main.cu $(DIRS)/clahe.cu
HEADERS = $(DIRS)/clahe.h 

# Directories
IMAGES_DIR = Images
GOLDEN_DIR = golden
OUTPUT_DIR = output

# Build target
all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(CFLAGS) $(SOURCES) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -f $(OUTPUT_DIR)/*.pgm