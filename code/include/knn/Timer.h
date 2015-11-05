#pragma once

#include <chrono>

class Timer {

public:
  inline Timer();

  inline void Start();

  inline double Stop();

  inline double Elapsed() const;

private:
  std::chrono::time_point<std::chrono::system_clock> start_;
  double elapsed_{};
}; // End class Timer

Timer::Timer() : start_(std::chrono::system_clock::now()) {}

void Timer::Start() { start_ = std::chrono::system_clock::now(); }

double Timer::Stop() {
  elapsed_ = 1e-9 *
             std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::system_clock::now() - start_)
                 .count();
  return elapsed_;
}

double Timer::Elapsed() const { return elapsed_; }
