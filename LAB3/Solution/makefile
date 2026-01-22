TARGET = Convolution2D_2 Convolution2D_4 Convolution2D_6a Convolution2D_6b Convolution2D_8
SRC_DIR = src
TARGET_DIR = target
CC = nvcc
CFLAGS = -O4 -I.
H_FILES = gputimer.h
# Build executables inside target/
EXES = $(TARGET:%=$(TARGET_DIR)/%)

all: $(EXES) $(H_FILES)

# Rule to build target/<name> from src/<name>.cu
$(TARGET_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) $< -o $@ 

clean:
	rm -rf $(TARGET_DIR)
