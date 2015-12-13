#pragma once

#include <string>
#include <map>

class ParseArguments {

public:
  ParseArguments(int argc, char const *const *argv);
  ~ParseArguments();

  bool operator()(std::string const &arg, std::string &value);
  bool operator()(std::string const &arg, int &value);
  bool operator()(std::string const &arg, float &value);

private:
  bool GetArg(std::string argName, std::string &argVal);

  std::map<std::string, std::string> args_;

};
