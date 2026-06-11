PROJECTNAME          = IvyFramework
PACKAGENAME          = IvyHeterogeneousCore


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

ifneq ($(strip $(USE_CUDA)),)
CXX           = nvcc
OPENMP_FLAGS  = 
CXXDEFINES    += -D__USE_CUDA__
NVLINKOPTS    = -Xnvlink --suppress-stack-size-warning
CXXFLAGS      = -arch=sm_86 -dc -rdc=true -x cu --cudart=shared $(NVLINKOPTS) -Xcompiler -fPIC -g -ftemplate-backtrace-limit=0 $(CXXOPTIM) $(CXXVEROPT) $(OPENMP_FLAGS) $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
EXEFLAGS      = $(filter-out -dc, $(CXXFLAGS))
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


.PHONY: all utests help compile clean
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
	rm -rf $(SANTESTEXEDIR)
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


include $(DEPS)

# ---------------------------------------------------------------------------
# Sanitizer and Valgrind test targets (host/g++ only; not for USE_CUDA builds)
#
#   make utests-asan      Build+run all unit tests with AddressSanitizer+UBSan
#   make utests-tsan      Build+run all unit tests with ThreadSanitizer
#   make run-utests       Build (make utests) then run every unit test
#   make valgrind-utests  Build (make utests) then run every test under Valgrind
#
# The Valgrind gate is leak- and memory-safety-focused (leaks, invalid reads/
# writes, use-after-free); Memcheck's uninitialised-value reporting is disabled
# (--undef-value-errors=no) to avoid failing on a pre-existing, unrelated
# benchmark issue (copy_data leaving the int target unpopulated in CPU mode).
# ---------------------------------------------------------------------------
SANTESTEXEDIR = $(COMPILEPATH)test_executables_san/
ASAN_FLAGS    = -fsanitize=address,undefined -fno-omit-frame-pointer
TSAN_FLAGS    = -fsanitize=thread -fno-omit-frame-pointer

# The autodiff unit test is out of scope for the memory/thread sanitizer gates:
# it has pre-existing leaks under ASan that are unrelated to core STL/memory code.
SANTESTSCC    = $(filter-out %utest_autodiff_basic_blocks.cc, $(TESTSCC))
VGTESTEXES    = $(filter-out %utest_autodiff_basic_blocks, $(TESTEXES))
# ThreadSanitizer is gated on the std::thread-based regression tests only.
# TSan cannot observe libgomp's internal synchronization, so it reports false
# races inside OpenMP runtime code (reductions/barriers) for the OpenMP-based
# element-loop tests; those are covered by ASan + Valgrind instead.
TSANTESTSCC   = $(wildcard $(TESTDIR)utest_unified_ptr_threads.cc)

.PHONY: utests-asan utests-tsan run-utests valgrind-utests santestexedirs

santestexedirs:
	mkdir -p $(SANTESTEXEDIR)

utests-asan: santestexedirs
	set -e; \
	for src in $(SANTESTSCC); do \
	  exe=$(SANTESTEXEDIR)$$(basename $$src .cc); \
	  echo "[ASan/UBSan] Compiling $$src"; \
	  $(CXX) $(EXEFLAGS) $(ASAN_FLAGS) -o $$exe $$src $(LIBS); \
	done; \
	for src in $(SANTESTSCC); do \
	  exe=$(SANTESTEXEDIR)$$(basename $$src .cc); \
	  echo "[ASan/UBSan] Running $$exe"; \
	  ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 $$exe > /dev/null; \
	done

utests-tsan: santestexedirs
	set -e; \
	for src in $(TSANTESTSCC); do \
	  exe=$(SANTESTEXEDIR)$$(basename $$src .cc).tsan; \
	  echo "[TSan] Compiling $$src"; \
	  $(CXX) $(EXEFLAGS) $(TSAN_FLAGS) -o $$exe $$src $(LIBS); \
	done; \
	for src in $(TSANTESTSCC); do \
	  exe=$(SANTESTEXEDIR)$$(basename $$src .cc).tsan; \
	  echo "[TSan] Running $$exe"; \
	  TSAN_OPTIONS=halt_on_error=1 $$exe > /dev/null; \
	done

run-utests: utests
	set -e; \
	for exe in $(TESTEXES); do \
	  echo "Running $$exe"; \
	  $$exe > /dev/null; \
	done

valgrind-utests: utests
	set -e; \
	for exe in $(VGTESTEXES); do \
	  echo "[Valgrind] Running $$exe"; \
	  valgrind --leak-check=full --errors-for-leak-kinds=definite,indirect \
	    --undef-value-errors=no --error-exitcode=1 $$exe > /dev/null; \
	done
