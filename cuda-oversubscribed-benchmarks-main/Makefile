
.PHONY: clean base

MAKE_ACTION = 
MAKE_FLAGS = -B

base:
	make $(MAKE_FLAGS) -C synthetic_benchmarks $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C nvidia-samples/sgemm/ $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C polybench/2DCONV $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C polybench/MVT $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C polybench/2MM $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C polybench/BICG $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C polybench/GEMM $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C synthetic_benchmarks/ $(MAKE_ACTION)
	make $(MAKE_FLAGS) -C rodinia/nw $(MAKE_ACTION)


clean: MAKE_ACTION += clean
clean: base
	rm -rf polybench/*/*exe
	rm -rf polybench/*/a.out
	rm -rf rapids/__pycache__
