#pragma once
#include <string>
#include <utility>

namespace tfs {
enum class ErrorCode { kOk=0, kInvalidArgument, kUnsupported, kIoError, kRuntimeError, kNotImplemented };
class Status {
 public:
  Status() = default;
  Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}
  static Status Ok() { return {}; }
  static Status InvalidArgument(std::string m) { return {ErrorCode::kInvalidArgument,std::move(m)}; }
  static Status Unsupported(std::string m) { return {ErrorCode::kUnsupported,std::move(m)}; }
  static Status IoError(std::string m) { return {ErrorCode::kIoError,std::move(m)}; }
  static Status RuntimeError(std::string m) { return {ErrorCode::kRuntimeError,std::move(m)}; }
  static Status NotImplemented(std::string m) { return {ErrorCode::kNotImplemented,std::move(m)}; }
  bool ok() const noexcept { return code_==ErrorCode::kOk; }
  ErrorCode code() const noexcept { return code_; }
  const std::string& message() const noexcept { return message_; }
 private: ErrorCode code_=ErrorCode::kOk; std::string message_;
};
}

