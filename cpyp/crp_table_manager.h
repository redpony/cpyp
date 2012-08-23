#ifndef _CPYP_CRP_TABLE_MANAGER_H_
#define _CPYP_CRP_TABLE_MANAGER_H_

#include <iostream>
#include <utility>
#include "sparse_vector.h"
#include "random.h"

namespace cpyp {

// these are helper classes for implementing token-based CRP samplers
// basically the data structures recommended by Blunsom et al. in the Note.

struct crp_histogram {
  //typedef std::map<unsigned, unsigned> MAPTYPE;
  typedef SparseVector<unsigned> MAPTYPE;
  typedef MAPTYPE::const_iterator const_iterator;

  inline void increment(unsigned bin, unsigned delta = 1u) {
    data[bin] += delta;
  }
  inline void decrement(unsigned bin, unsigned delta = 1u) {
    unsigned r = data[bin] -= delta;
    if (!r) data.erase(bin);
  }
  inline void move(unsigned from_bin, unsigned to_bin, unsigned delta = 1u) {
    decrement(from_bin, delta);
    increment(to_bin, delta);
  }
  inline const_iterator begin() const { return data.begin(); }
  inline const_iterator end() const { return data.end(); }
  void swap(crp_histogram& other) {
    std::swap(data, other.data);
  }

 private:
  MAPTYPE data;
};

void swap(crp_histogram& a, crp_histogram& b) {
  a.swap(b);
}

// A crp_table_manager tracks statistics about all customers
// and tables serving some dish in a CRP and can correctly sample what
// table to remove a customer from and what table to join
struct crp_table_manager {
  crp_table_manager() : customers(), tables() {}

  inline unsigned num_tables() const {
    return tables;
  }

  inline unsigned num_customers() const {
    return customers;
  }

  inline void create_table() {
    h.increment(1);
    ++tables;
    ++customers;
  }

  // seat a customer at a table proportional to the number of customers seated at a table, less the discount
  // *new tables are never created by this function!
  // returns the number of customers already seated at the table (always > 0)
  template<typename Engine>
  unsigned share_table(const double discount, Engine& eng) {
    const double z = customers - discount * num_tables();
    double r = z * sample_uniform01<double>(eng);
    const auto it = [&] {
      const auto end = h.end();
      auto i = h.begin();
      for (; i != end; ++i) {
        const double thresh = (i->first - discount) * i->second;
        if (thresh > r) break;
        r -= thresh;
      }
      return i;
    }();
    const unsigned cc = it->first;
    h.move(cc, cc + 1);
    ++customers;
    return cc;
  }

  // randomly sample a customer
  // *tables may be removed
  // returns -1 if a table is removed, 0 otherwise
  template<typename Engine>
  inline int remove_customer(Engine& eng, unsigned* selected_table_postcount) {
    int r = sample_uniform01<double>(eng) * num_customers();
    const auto it = [&] {
      const auto end = h.end();
      auto i = h.begin();
      for (; i != end; ++i) {
        // sample randomly, i.e. *don't* discount
        const double thresh = i->first * i->second;
        if (thresh > r) break;
        r -= thresh;
      }
      return i;
    }();
    --customers;
    const unsigned tc = it->first;
    if (selected_table_postcount) *selected_table_postcount = tc - 1;
    if (tc == 1) { // remove customer from table containing a single customer
      h.decrement(1);
      --tables;
      return -1;
    } else {
      h.move(tc, tc - 1);
      return 0;
    }
  }

  typedef crp_histogram::const_iterator const_iterator;
  const_iterator begin() const { return h.begin(); }
  const_iterator end() const { return h.end(); }

  unsigned customers;
  unsigned tables;
  crp_histogram h;
};

void swap(crp_table_manager& a, crp_table_manager& b) {
  std::swap(a.customers, b.customers);
  std::swap(a.tables, b.tables);
  std::swap(a.h, b.h);
}

std::ostream& operator<<(std::ostream& os, const crp_table_manager& tm) {
  os << '[' << tm.num_customers() << " total customers at " << tm.num_tables() << " tables ||| ";
  bool first = true;
  for (auto& table : tm) {
    if (first) first = false; else os << "  --  ";
    os << '(' << table.first << ") x " << table.second;
  }
  return os << ']';
}

}

#endif
