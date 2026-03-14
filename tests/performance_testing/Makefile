NVCC      = nvcc
NVCCFLAGS = -O2 -lm

TARGETS = perf_dirty_tracking

all: $(TARGETS)

perf_dirty_tracking: perf_dirty_tracking.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
