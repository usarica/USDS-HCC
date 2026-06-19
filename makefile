PROJECTNAME          = IvyFramework
PACKAGENAME          = IvyHCC

# Parallel compilation: use all available cores.
# -k (keep-going) ensures that a single failing TU does not abort other independent targets.
MAKEFLAGS += -j$(shell nproc) -k


COMPILEPATH          = $(PWD)/

INCLUDEDIR           = $(COMPILEPATH)interface/
BINDIR               = $(COMPILEPATH)bin/
EXEDIR               = $(COMPILEPATH)executables/
TESTEXEDIR           = $(COMPILEPATH)test_executables/
TESTDIR              = $(COMPILEPATH)test/
RUNDIR               = $(COMPILEPATH)


EXTCXXFLAGS   =
EXTLIBS       =

ROOTCFLAGS    =
ROOTLIBS      =

CXX           = g++
CXXINC        = -I$(INCLUDEDIR)
CXXDEFINES    =
CXXVEROPT     = -std=c++20
CXXOPTIM      = -O2

# Check for OpenMP support
OPENMP_TEST := $(shell echo '\#include <omp.h>' | $(CXX) -x c++ -fopenmp -E - > /dev/null 2>&1 && echo "yes")
ifeq ($(OPENMP_TEST),yes)
  OPENMP_FLAGS=-fopenmp
else
  OPENMP_FLAGS=
endif

CXXFLAGS      = -fPIC -g -ftemplate-backtrace-limit=0 $(CXXOPTIM) $(CXXVEROPT) $(OPENMP_FLAGS) $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
EXEFLAGS      = $(CXXFLAGS)

# Constrain default CUDA code generation to RTX A5000 class hardware (sm_86).
# Override only when explicitly requested, e.g. `GPU_ARCH_RAW=90 USE_CUDA=1 make utests`.
GPU_ARCH_RAW := $(shell command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' || echo "")
ifeq ($(strip $(GPU_ARCH_RAW)),)
  GPU_ARCH_RAW := 86
endif
GPU_GENCODE := -gencode arch=compute_$(GPU_ARCH_RAW),code=[sm_$(GPU_ARCH_RAW),compute_$(GPU_ARCH_RAW)]

ifneq ($(strip $(USE_CUDA)),)
CXX           = nvcc
OPENMP_FLAGS  =
CXXDEFINES    += -D__USE_CUDA__
NVLINKOPTS    = -Xnvlink --suppress-stack-size-warning
CXXFLAGS      = $(GPU_GENCODE) -dc -rdc=true -x cu --cudart=shared $(NVLINKOPTS) -Xcompiler -fPIC -g -ftemplate-backtrace-limit=0 $(CXXOPTIM) $(CXXVEROPT) $(OPENMP_FLAGS) $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
EXEFLAGS      = $(filter-out -dc, $(CXXFLAGS))
LIBDLINKFLAGS = $(GPU_GENCODE) $(NVLINKOPTS) -Xcompiler -fPIC
endif

ifeq ($(strip $(USE_CUDA)),)
  # Precompiled header — g++ only. nvcc does not support .gch (nvcc --pch is fatal).
  PCHFILE    = $(INCLUDEDIR)IvyPrecompiledHeader.h
  PCHOUT     = $(PCHFILE).gch
  # EXEFLAGS = $(CXXFLAGS) is a lazy recursive assignment (line 40), so adding
  # -include to CXXFLAGS automatically propagates to EXEFLAGS; no duplicate needed.
  CXXFLAGS  += -include $(PCHFILE)
endif

NLIBS        += $(EXTLIBS)
# Filter out libNew because it leads to floating-point exceptions in versions of ROOT prior to 6.08.02
# See the thread https://root-forum.cern.ch/t/linking-new-library-leads-to-a-floating-point-exception-at-startup/22404
LIBS          = $(filter-out -lNew, $(NLIBS))

BINSCC = $(wildcard $(BINDIR)*.cc)
BINSCXX = $(wildcard $(BINDIR)*.cxx)
TESTSCC = $(wildcard $(TESTDIR)*.cc)
TESTSCXX = $(wildcard $(TESTDIR)*.cxx)
EXESPRIM = $(BINSCC:.cc=) $(BINSCXX:.cxx=)
TESTEXESPRIM = $(TESTSCC:.cc=) $(TESTSCXX:.cxx=)
EXES = $(subst $(BINDIR),$(EXEDIR),$(EXESPRIM))
TESTEXES = $(subst $(TESTDIR),$(TESTEXEDIR),$(TESTEXESPRIM))

STRICT_LOG_DIR ?= $(COMPILEPATH)strict_logs
STRICT_WARNING_PATTERN ?= warning|Warning|\#20011|\#20012|\#20014
STRICT_ASAN_FLAGS ?= -Wall -Wextra -Wpedantic -Wconversion -Werror -fsanitize=address,leak -fno-omit-frame-pointer


.PHONY: all utests help compile clean lib pch distclean diagnostics-baseline diagnostics-cpu diagnostics-cuda warning-inventory
.SILENT: exedirs testexedirs clean $(EXES) $(TESTEXES)


all: $(EXES)


utests: $(TESTEXES)


exedirs:
	mkdir -p $(EXEDIR)


testexedirs:
	mkdir -p $(TESTEXEDIR)


alldirs: exedirs testexedirs


$(EXEDIR)%:: $(BINDIR)%.cc | exedirs
	echo "Compiling $<"; \
	$(CXX) $(EXEFLAGS) -o $@ $< $(LIBS)


$(TESTEXEDIR)%:: $(TESTDIR)%.cc | testexedirs
	echo "Compiling $<"; \
	$(CXX) $(EXEFLAGS) -o $@ $< $(LIBS)


clean:
	rm -rf $(EXEDIR)
	rm -rf $(TESTEXEDIR)
	rm -rf $(LIBDIR)
	rm -f $(INCLUDEDIR)/*.gch
	rm -f $(BINDIR)*.o
	rm -f $(BINDIR)*.so
	rm -f $(BINDIR)*.d
	rm -rf $(RUNDIR)Pdfdata
	rm -f $(RUNDIR)*.DAT
	rm -f $(RUNDIR)*.dat
	rm -f $(RUNDIR)br.sm*
	rm -f $(RUNDIR)*.cc
	rm -f $(RUNDIR)*.o
	rm -f $(RUNDIR)*.so
	rm -f $(RUNDIR)*.d
	rm -f $(RUNDIR)*.pcm
	rm -f $(RUNDIR)*.pyc
	rm -rf $(TESTDIR)Pdfdata
	rm -f $(TESTDIR)*.DAT
	rm -f $(TESTDIR)*.dat
	rm -f $(TESTDIR)br.sm*
	rm -f $(TESTDIR)*.o
	rm -f $(TESTDIR)*.so
	rm -f $(TESTDIR)*.d
	rm -f $(TESTDIR)*.pcm
	rm -f $(TESTDIR)*.pyc


LIBDIR         = $(COMPILEPATH)lib/
LIBNAME        = libIvyHCC
SRCDIR         = $(COMPILEPATH)src/

lib:
	mkdir -p $(LIBDIR)
ifeq ($(strip $(USE_CUDA)),)
	$(CXX) $(CXXFLAGS) -shared -o $(LIBDIR)$(LIBNAME).so $(SRCDIR)IvyHCC.cc
else
	$(CXX) $(CXXFLAGS) -o $(LIBDIR)$(LIBNAME).o $(SRCDIR)IvyHCC.cc
	$(CXX) $(LIBDLINKFLAGS) -dlink -shared $(LIBDIR)$(LIBNAME).o -o $(LIBDIR)$(LIBNAME)_dlink.o
	$(CXX) $(LIBDLINKFLAGS) -shared $(LIBDIR)$(LIBNAME).o $(LIBDIR)$(LIBNAME)_dlink.o -o $(LIBDIR)$(LIBNAME).so -lcudart
	rm -f $(LIBDIR)$(LIBNAME).o $(LIBDIR)$(LIBNAME)_dlink.o
endif


pch:
ifeq ($(strip $(USE_CUDA)),)
	@echo "Building g++ precompiled header (CPU only)..."
	$(CXX) $(filter-out -include $(PCHFILE), $(CXXFLAGS)) -x c++-header -o $(PCHOUT) $(PCHFILE)
	@echo "PCH written: $(PCHOUT)"
else
	@echo "PCH skipped: nvcc does not support .gch precompiled headers."
endif

distclean: clean
	@echo "Removing precompiled header artifacts..."
	rm -f $(INCLUDEDIR)*.gch

diagnostics-cpu:
	mkdir -p $(STRICT_LOG_DIR)
	printf "=== CPU clean build with timing ===\n" > $(STRICT_LOG_DIR)/10_cpu_clean_build.log
	$(MAKE) distclean >> $(STRICT_LOG_DIR)/10_cpu_clean_build.log 2>&1
	start=$$(date +%s); \
	$(MAKE) pch utests >> $(STRICT_LOG_DIR)/10_cpu_clean_build.log 2>&1; \
	end=$$(date +%s); \
	printf "real_seconds=%s\n" $$((end-start)) >> $(STRICT_LOG_DIR)/10_cpu_clean_build.log
	printf "=== CPU incremental build with timing ===\n" > $(STRICT_LOG_DIR)/11_cpu_incremental_build.log
	start=$$(date +%s); \
	$(MAKE) utests >> $(STRICT_LOG_DIR)/11_cpu_incremental_build.log 2>&1; \
	end=$$(date +%s); \
	printf "real_seconds=%s\n" $$((end-start)) >> $(STRICT_LOG_DIR)/11_cpu_incremental_build.log
	printf "=== CPU test run ===\n" > $(STRICT_LOG_DIR)/12_cpu_tests.log
	found=0; \
	for t in ./test_executables/utest_*; do \
		if [ -x "$$t" ]; then \
			found=1; \
			printf "[RUN] %s\n" "$$t" >> $(STRICT_LOG_DIR)/12_cpu_tests.log; \
			"$$t" >> $(STRICT_LOG_DIR)/12_cpu_tests.log 2>&1; \
		fi; \
	done; \
	if [ "$$found" -eq 0 ]; then \
		printf "No CPU test executables found\n" >> $(STRICT_LOG_DIR)/12_cpu_tests.log; \
	fi
	printf "=== CPU ASAN/LSAN build+test ===\n" > $(STRICT_LOG_DIR)/13_cpu_asan_lsan.log
	$(MAKE) distclean >> $(STRICT_LOG_DIR)/13_cpu_asan_lsan.log 2>&1
	start=$$(date +%s); \
	$(MAKE) utests EXTCXXFLAGS="$(STRICT_ASAN_FLAGS)" >> $(STRICT_LOG_DIR)/13_cpu_asan_lsan.log 2>&1; \
	end=$$(date +%s); \
	printf "real_seconds=%s\n" $$((end-start)) >> $(STRICT_LOG_DIR)/13_cpu_asan_lsan.log
	found=0; \
	for t in ./test_executables/utest_*; do \
		if [ -x "$$t" ]; then \
			found=1; \
			printf "[ASAN-RUN] %s\n" "$$t" >> $(STRICT_LOG_DIR)/13_cpu_asan_lsan.log; \
			ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 LSAN_OPTIONS=verbosity=1:log_threads=1 "$$t" >> $(STRICT_LOG_DIR)/13_cpu_asan_lsan.log 2>&1; \
		fi; \
	done; \
	if [ "$$found" -eq 0 ]; then \
		printf "No ASAN test executables found\n" >> $(STRICT_LOG_DIR)/13_cpu_asan_lsan.log; \
	fi

diagnostics-cuda:
	mkdir -p $(STRICT_LOG_DIR)
	printf "=== CUDA availability ===\n" > $(STRICT_LOG_DIR)/20_cuda_env.log
	command -v nvcc >> $(STRICT_LOG_DIR)/20_cuda_env.log 2>&1 || true
	command -v compute-sanitizer >> $(STRICT_LOG_DIR)/20_cuda_env.log 2>&1 || true
	nvidia-smi >> $(STRICT_LOG_DIR)/20_cuda_env.log 2>&1 || true
	printf "=== CUDA clean build with timing ===\n" > $(STRICT_LOG_DIR)/21_cuda_clean_build.log
	if command -v nvcc >/dev/null 2>&1; then \
		$(MAKE) distclean >> $(STRICT_LOG_DIR)/21_cuda_clean_build.log 2>&1; \
		start=$$(date +%s); \
		env USE_CUDA=1 $(MAKE) utests >> $(STRICT_LOG_DIR)/21_cuda_clean_build.log 2>&1; \
		end=$$(date +%s); \
		printf "real_seconds=%s\n" $$((end-start)) >> $(STRICT_LOG_DIR)/21_cuda_clean_build.log; \
	else \
		printf "nvcc not found; CUDA clean build skipped\n" >> $(STRICT_LOG_DIR)/21_cuda_clean_build.log; \
	fi
	printf "=== CUDA incremental build with timing ===\n" > $(STRICT_LOG_DIR)/22_cuda_incremental_build.log
	if command -v nvcc >/dev/null 2>&1; then \
		start=$$(date +%s); \
		env USE_CUDA=1 $(MAKE) utests >> $(STRICT_LOG_DIR)/22_cuda_incremental_build.log 2>&1; \
		end=$$(date +%s); \
		printf "real_seconds=%s\n" $$((end-start)) >> $(STRICT_LOG_DIR)/22_cuda_incremental_build.log; \
	else \
		printf "nvcc not found; CUDA incremental build skipped\n" >> $(STRICT_LOG_DIR)/22_cuda_incremental_build.log; \
	fi
	printf "=== CUDA test run ===\n" > $(STRICT_LOG_DIR)/23_cuda_tests.log
	if command -v nvcc >/dev/null 2>&1; then \
		found=0; \
		for t in ./test_executables/utest_*; do \
			if [ -x "$$t" ]; then \
				found=1; \
				printf "[CUDA-RUN] %s\n" "$$t" >> $(STRICT_LOG_DIR)/23_cuda_tests.log; \
				"$$t" >> $(STRICT_LOG_DIR)/23_cuda_tests.log 2>&1; \
			fi; \
		done; \
		if [ "$$found" -eq 0 ]; then \
			printf "No CUDA test executables found\n" >> $(STRICT_LOG_DIR)/23_cuda_tests.log; \
		fi; \
	else \
		printf "nvcc not found; CUDA test run skipped\n" >> $(STRICT_LOG_DIR)/23_cuda_tests.log; \
	fi
	printf "=== compute-sanitizer memcheck ===\n" > $(STRICT_LOG_DIR)/24_cuda_memcheck.log
	if command -v nvcc >/dev/null 2>&1 && command -v compute-sanitizer >/dev/null 2>&1; then \
		found=0; \
		for t in ./test_executables/utest_*; do \
			if [ -x "$$t" ]; then \
				found=1; \
				printf "[MEMCHECK] %s\n" "$$t" >> $(STRICT_LOG_DIR)/24_cuda_memcheck.log; \
				compute-sanitizer --tool memcheck --leak-check full "$$t" >> $(STRICT_LOG_DIR)/24_cuda_memcheck.log 2>&1; \
			fi; \
		done; \
		if [ "$$found" -eq 0 ]; then \
			printf "No binaries for memcheck\n" >> $(STRICT_LOG_DIR)/24_cuda_memcheck.log; \
		fi; \
	else \
		printf "compute-sanitizer and/or nvcc not found; memcheck skipped\n" >> $(STRICT_LOG_DIR)/24_cuda_memcheck.log; \
	fi

warning-inventory:
	mkdir -p $(STRICT_LOG_DIR)
	printf "=== Warning inventory ===\n" > $(STRICT_LOG_DIR)/30_warning_inventory.log
	for f in $(STRICT_LOG_DIR)/*.log; do \
		[ "$$f" = "$(STRICT_LOG_DIR)/30_warning_inventory.log" ] && continue; \
		c=$$(grep -c -E "$(STRICT_WARNING_PATTERN)" "$$f" || true); \
		printf "%s: warnings=%s\n" "$$f" "$$c" >> $(STRICT_LOG_DIR)/30_warning_inventory.log; \
	done
	printf "%s\n" "--- __host__/__device__ related hits ---" >> $(STRICT_LOG_DIR)/30_warning_inventory.log
	for f in $(STRICT_LOG_DIR)/*.log; do \
		[ "$$f" = "$(STRICT_LOG_DIR)/30_warning_inventory.log" ] && continue; \
		grep -n -E "__host__|__device__|#20011|#20012|#20014" "$$f" >> $(STRICT_LOG_DIR)/30_warning_inventory.log || true; \
	done

diagnostics-baseline: diagnostics-cpu diagnostics-cuda warning-inventory
	@echo "Diagnostics complete under $(STRICT_LOG_DIR)"

include $(DEPS)
