NVCC      = nvcc
NVCCFLAGS = -O2 -lm

SRCS = $(wildcard *.cu)
BINS = $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	-$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

clean_csv:
	rm -f *.csv

.PHONY: all clean
