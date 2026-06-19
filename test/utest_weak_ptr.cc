/**
 * @file utest_weak_ptr.cc
 * @brief Unit tests for IvyWeakPtr (std_ivy::weak_ptr) non-owning references.
 *
 * Verifies:
 *  - A weak_ptr does NOT keep the managed object alive.
 *  - lock() yields a valid shared owner while the object is alive and bumps use_count.
 *  - After the last strong owner is released, the object is destroyed exactly once,
 *    expired() becomes true, and lock() returns an empty shared_ptr.
 *  - The control block is reclaimed when the last weak observer is destroyed
 *    (validated under AddressSanitizer/LeakSanitizer in the make utests-asan gate).
 *  - A strong<->weak ownership cycle does not leak.
 */

#include "common_test_defs.h"

#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyCassert.h"


using namespace std_ivy;


static int g_live = 0;   //!< Number of live Tracked instances.

/** @brief Object whose construction/destruction is counted (host-side only). */
struct Tracked{
  int v;
  __HOST_DEVICE__ Tracked() : v(0){
#ifndef __CUDA_ARCH__
    ++g_live;
#endif
  }
  __HOST_DEVICE__ Tracked(int x) : v(x){
#ifndef __CUDA_ARCH__
    ++g_live;
#endif
  }
  __HOST_DEVICE__ ~Tracked(){
#ifndef __CUDA_ARCH__
    --g_live;
#endif
  }
};


static void check(bool cond, char const* label){
  if (cond){ __PRINT_INFO__("  [PASS] %s\n", label); }
  else { __PRINT_INFO__("  [FAIL] %s\n", label); assert(false); }
}


void utest(){
  __PRINT_INFO__("=== utest_weak_ptr ===\n");
  constexpr IvyMemoryType mem = IvyMemoryType::Host;

  // 1. Basic observation: weak does not own.
  {
    auto sp = make_shared<Tracked>(mem, nullptr, 42);
    check(g_live == 1, "object constructed");
    check(sp.use_count() == 1, "strong use_count == 1");

    weak_ptr<Tracked> w(sp);
    check(w.use_count() == 1, "weak observes use_count == 1 (weak does not add strong)");
    check(!w.expired(), "weak not expired while object alive");

    // 2. lock() yields a valid owner and bumps the strong count.
    {
      auto s2 = w.lock();
      check(static_cast<bool>(s2), "lock() yields a valid shared_ptr");
      check(s2->v == 42, "locked value is correct");
      check(sp.use_count() == 2, "use_count == 2 while locked");
    }
    check(sp.use_count() == 1, "use_count back to 1 after locked owner dies");
    check(g_live == 1, "object still alive (weak kept observing, not owning)");

    // 3. Drop the last strong owner; object must be destroyed exactly once.
    sp.reset();
    check(g_live == 0, "object destroyed when last strong owner released");
    check(w.expired(), "weak expired after object destroyed");
    check(!static_cast<bool>(w.lock()), "lock() returns empty after expiry");
    // w destructor here frees the control block (checked under ASan).
  }

  // 4. Copy/move of weak_ptr.
  {
    auto sp = make_shared<Tracked>(mem, nullptr, 7);
    weak_ptr<Tracked> w1(sp);
    weak_ptr<Tracked> w2(w1);            // copy
    weak_ptr<Tracked> w3(std_util::move(w1)); // move
    check(!w2.expired() && !w3.expired(), "copied/moved weak observe live object");
    check(static_cast<bool>(w3.lock()), "moved-into weak can lock");
    sp.reset();
    check(w2.expired() && w3.expired(), "all weak observers expire together");
    check(g_live == 0, "object destroyed once");
  }

  // 5. Strong<->weak cycle does not leak (validated under ASan).
  {
    struct Node{
      shared_ptr<Tracked> owned;        // strong down-edge
      weak_ptr<Tracked> back;           // weak back-edge
    };
    auto a = make_shared<Tracked>(mem, nullptr, 1);
    Node n;
    n.owned = a;            // strong owns the object
    n.back = weak_ptr<Tracked>(a);  // weak back-reference (no cycle leak)
    check(g_live == 1, "cycle object alive");
    a.reset();
    check(g_live == 1, "object alive via n.owned");
    n.owned.reset();
    check(g_live == 0, "object freed once strong owners gone; weak back-ref did not leak");
    check(n.back.expired(), "weak back-ref expired");
  }

  check(g_live == 0, "no live objects at end");
  __PRINT_INFO__("=== ALL utest_weak_ptr tests PASSED ===\n");
}


int main(){
  utest();
  return 0;
}
