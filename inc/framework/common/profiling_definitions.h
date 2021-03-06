/**
* Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
* Copyright 2021, 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AIR_CXX_PROFILING_DEFINITIONS_H
#define AIR_CXX_PROFILING_DEFINITIONS_H
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include "graph/profiler.h"
#include "external/ge/ge_api_types.h"
#include "toolchain/prof_callback.h"
namespace ge {
namespace profiling {
enum {
  kAclCompileAndExecute,
  kAclMatchOpModel,
  kAclMatchStaticOpModel,
  kAclMatchDynamicOpModel,
  kAclExecuteAsync,
  kAclLoadSingleOp,
  kAclBuildOpModel,
  kInferShape,
  kTiling,
  kUpdateShape,
  kConstPrepare,
  kInitHybridExecuteArgs,
  kInitInferShapeContext,
  kDestroyInferShapeContext,
  kResetSubgraphExecutor,
  kCommitInferShapeTask,
  kDeviceToHost,
  kPrepareTask,
  kLaunchTask,
  kCommitTilingTask,
  kAtomic,
  kKernelLaunchPrepare,
  kRtKernelLaunch,
  kRtEventCreateRecord,
  kRtEventSync,
  kRtEventDestroy,
  kRtStreamSync,
  kOpExecute,
  kModelExecute,
  kAllocMem,
  kCopyH2D,
  kPrepareNode,
  kWaitForPrepareDone,
  kPropgateOutputs,
  kOnNodeDoneCallback,
  kValidateInputTensor,
  kAfterExecuted,
  kRtEventSychronize,
  kInferShapeWaitDependShape,
  kInferShapeWaitInputTensor,
  kInferShapeCallInferFunc,
  kInferShapePropgate,
  // v2 control node
  kSelectBranch,
  kExecuteSubGraph,
  kInitSubGraphExecutor,
  // fuzz compile
  kSelectBin,
  kFindCompileCache,
  kAddCompileCache,
  kFuzzCompileOp,
  kCalcRuningParam,
  kGenTask,
  kRegisterBin,

  // Add new definitions here
  kProfilingIndexEnd
};
constexpr uint64_t kInvalidHashId = 0UL;

class ProfilingContext {
 public:
  static bool IsDumpToStdEnabled();
  static ProfilingContext &GetInstance();
  ProfilingContext();
  ~ProfilingContext();

  /*
   * ?????????????????????`IsEnabled`?????????profiler_??????????????????????????????????????????enabled?????????????????????????????????????????????
   * ??????????????????????????????profiler_??????????????????profiling?????????????????????????????????
   * ?????????????????????profiling??????????????????????????????????????????`RegisterString`??????profiler_????????????????????????????????????????????????????????????index??????
   * ????????????????????????????????????????????????profiling????????????????????????????????????profiling???????????????????????????????????????????????????
   * ??????????????????????????????????????????????????????????????????????????????????????????profiling????????????????????????????????????????????????????????????
   */
  bool IsEnabled() const noexcept {
    return enabled_ && (profiler_ != nullptr);
  }
  void SetEnable() noexcept {
    enabled_ = true;
  }
  void SetDisable() noexcept {
    enabled_ = false;
  }

  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et,
                           const std::chrono::time_point<std::chrono::system_clock> time_point) {
    if (IsEnabled()) {
      profiler_->RecordCurrentThread(element, event, et, time_point);
    }
  }

  void RecordCurrentThread(const int64_t element, const int64_t event, const EventType et) {
    RecordCurrentThread(element, event, et, std::chrono::system_clock::now());
  }

  const Profiler *GetProfiler() const {
    return profiler_.get();
  }

  void Dump(std::ostream &out_stream) const {
    if (IsEnabled()) {
      profiler_->Dump(out_stream);
    } else {
      out_stream << "Profiling not enable, skip to dump" << std::endl;
    }
  }

  void DumpToStdOut() const {
    Dump(std::cout);
  }

  void Reset() {
    if (IsEnabled()) {
      profiler_->Reset();
    }
  }

  int64_t RegisterString(const std::string &str);
  int64_t RegisterStringHash(const uint64_t hash_id, const std::string &str);
  void UpdateElementHashId(const MsprofReporterCallback reporter_callback);
  static Status QueryHashId(const MsprofReporterCallback reporter_callback, const std::string &src_str,
                            uint64_t &hash_id);
  size_t GetRegisterStringNum() const {
    return strings_to_index_.size();
  }

  void Init();

 private:
  void UpdateHashByStr(const std::string &str, const uint64_t hash);

 private:
  bool inited_;
  bool enabled_;
  int64_t str_index_;
  std::unordered_map<std::string, int64_t> strings_to_index_;
  std::mutex strings_to_index_mutex_;
  std::unique_ptr<Profiler> profiler_;
};

class ScopeProfiler {
 public:
  ScopeProfiler(const int64_t element, const int64_t event) : element_(element), event_(event) {
    if (ProfilingContext::GetInstance().IsEnabled()) {
      start_trace_ = std::chrono::system_clock::now();
    }
  }
  ~ScopeProfiler() {
    if (ProfilingContext::GetInstance().IsEnabled()) {
      ProfilingContext::GetInstance().RecordCurrentThread(element_, event_, EventType::kEventStart, start_trace_);
      ProfilingContext::GetInstance().RecordCurrentThread(element_, event_, EventType::kEventEnd);
    }
  }
  void SetElement(const int64_t element) {
    element_ = element;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_trace_;
  int64_t element_;
  int64_t event_;
};
}  // namespace profiling
}  // namespace ge
#define PROFILING_START(element, event)                                                  \
  ge::profiling::ProfilingContext::GetInstance().RecordCurrentThread((element), (event), \
                                                                     ge::profiling::EventType::kEventStart)
#define PROFILING_END(element, event)                                                    \
  ge::profiling::ProfilingContext::GetInstance().RecordCurrentThread((element), (event), \
                                                                     ge::profiling::EventType::kEventEnd)
#define PROFILING_SCOPE(element, event) ge::profiling::ScopeProfiler profiler((element), (event))
#define PROFILING_SCOPE_ELEMENT(element) profiler.SetElement((element))
#endif  // AIR_CXX_PROFILING_DEFINITIONS_H
