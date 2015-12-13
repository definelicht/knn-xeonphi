#include "knn/ParseArguments.h"
#include <algorithm>
#include <iostream>
#include <regex>
#include <string>

ParseArguments::ParseArguments(int argc, char const *const *argv) : args_() {
  static const std::regex pattern("-([a-zA-Z_0-9]+)=([^ ]*)",
                                  std::regex_constants::basic);
  std::cmatch match;
  for (int i = 1; i < argc; ++i) {
    if (std::regex_match(argv[i], match, pattern) && match.size() == 3) {
      std::string argName(match[1]);
      std::transform(argName.begin(), argName.end(), argName.begin(),
                     ::tolower);
      args_.insert(std::make_pair(argName, match[2]));
    }
  }
}

ParseArguments::~ParseArguments() {
  for (auto &arg : args_) {
    std::cerr << "Warning: unused command line argument -" << arg.first << "="
              << arg.second << "\n";
  }
}

bool ParseArguments::operator()(std::string const &argName, int &output) {
  std::string argVal;
  if (!GetArg(argName, argVal)) {
    return false;
  }
  try {
    output = std::stoi(argVal);
  } catch (std::invalid_argument err) {
    return false;
  }
  return true;
}

bool ParseArguments::operator()(std::string const &argName, float &output) {
  std::string argVal;
  if (!GetArg(argName, argVal)) {
    return false;
  }
  try {
    output = std::stof(argVal);
  } catch (std::invalid_argument err) {
    return false;
  }
  return true;
}

bool ParseArguments::operator()(std::string const &argName,
                                std::string &output) {
  if (GetArg(argName, output)) {
    return true;
  }
  return false;
}

bool ParseArguments::GetArg(std::string argName, std::string &argVal) {
  std::transform(argName.cbegin(), argName.cend(), argName.begin(), ::tolower);
  auto arg = args_.find(argName);
  if (arg == args_.end()) {
    return false;
  }
  argVal = std::move(arg->second);
  args_.erase(arg);
  return true;
}
