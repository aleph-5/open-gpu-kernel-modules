NVCC     = nvcc
NVCCFLAGS =

TARGETS = test_dirty_tracking test_dirty_tracking_suite

all: $(TARGETS)

test_dirty_tracking: test_dirty_tracking.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

test_dirty_tracking_suite: test_dirty_tracking_suite.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean

