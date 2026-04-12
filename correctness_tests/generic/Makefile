NVCC      = /usr/local/cuda/bin/nvcc
NVCCFLAGS =

TARGETS = test_dirty_tracking_suite

all: $(TARGETS)

test_dirty_tracking_suite: test_dirty_tracking_suite.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

