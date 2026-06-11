#include "common_test_defs.h"

#include "std_ivy/IvyChrono.h"
#include "std_ivy/IvyAlgorithm.h"


using std_ivy::IvyMemoryType;


template<unsigned int nsum, unsigned int nsum_serial> void utest(IvyGPUStream& stream, double* sum_vals){
  __PRINT_INFO__("|*** Benchmarking parallel and serial summation... ***|\n");

  auto time_sum_s = std_chrono::high_resolution_clock::now();
  __PRINT_INFO__("Calling add_serial...\n");
  double sum_s = std_algo::add_serial<double>(sum_vals, nsum);
  __PRINT_INFO__("add_serial is complete.\n");
  auto time_sum_s_end = std_chrono::high_resolution_clock::now();
  auto time_sum_s_ms = std_chrono::duration_cast<std_chrono::microseconds>(time_sum_s_end - time_sum_s).count()/1000.;
  __PRINT_INFO__("\t- Sum serial time = %f ms\n", time_sum_s_ms);
  __PRINT_INFO__("\t- Sum serial = %f\n", sum_s);

  IvyTypes::size_t neff=0;
  std_algo::add_parallel_op<double>::parallel_op_n_mem(nsum, nsum_serial, neff);
  __PRINT_INFO__("\t- n: %i, neff: %i\n", nsum, neff);

  IvyGPUEvent ev_sum(IvyGPUEvent::EventFlags::Default); ev_sum.record(stream);
  __PRINT_INFO__("Starting event is recorded.\n");
  __PRINT_INFO__("Calling add_parallel...\n");
  double sum_p = std_algo::add_parallel<double>(sum_vals, nsum, nsum_serial, IvyMemoryType::Host, stream);
  __PRINT_INFO__("add_parallel is complete.\n");
  IvyGPUEvent ev_sum_end(IvyGPUEvent::EventFlags::Default); ev_sum_end.record(stream);
  ev_sum_end.synchronize();
  __PRINT_INFO__("Ending event is recorded and synchronized.\n");
  auto time_sum = ev_sum_end.elapsed_time(ev_sum);
  __PRINT_INFO__("\t- Sum parallel time = %f ms\n", time_sum);
  __PRINT_INFO__("\t- Sum parallel = %f\n", sum_p);

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}


int main(){
  constexpr unsigned char nStreams = 3;

  IvyGPUStream* streams[nStreams]{
    IvyStreamUtils::make_global_gpu_stream(),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking)
  };

  constexpr unsigned int nsum = 1000000;
  constexpr unsigned int nsum_serial = 64;
  double sum_vals[nsum];
  for (unsigned int i = 0; i < nsum; i++) sum_vals[i] = i+1;

  for (unsigned char i = 0; i < nStreams; i++){
    __PRINT_INFO__("**********\n");

    auto& stream = *(streams[i]);
    __PRINT_INFO__("Stream %i (%p, %p, size in bytes = %llu) computing...\n", i, &stream, (void*) &stream.stream(), __STATIC_CAST__(unsigned long long, sizeof(&stream)));

    utest<nsum, nsum_serial>(stream, sum_vals);

    __PRINT_INFO__("**********\n");
  }

  // Clean up local streams
  for (unsigned char i = 0; i < nStreams; i++) IvyStreamUtils::destroy_stream(streams[i]);

  return 0;
}
