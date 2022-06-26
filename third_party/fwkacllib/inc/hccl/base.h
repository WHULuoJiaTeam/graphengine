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

/**
 * @file base.h
 * @brief HCOM data type definition
 *
 */

#ifndef HCCL_BASE_H_
#define HCCL_BASE_H_
#include <hccl/hccl_types.h>
#include <string>
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

/**
 * @brief Horovod Reduction opperation
 */
typedef enum {
    HOROVOD_REDUCE_AVERAGE = 0, /**< average */
    HOROVOD_REDUCE_SUM = 1,     /**< sum */
    HOROVOD_REDUCE_ADASUM = 2,  /**< adasum */
    HOROVOD_REDUCE_MIN = 3,     /**< min */
    HOROVOD_REDUCE_MAX = 4,     /**< max */
    HOROVOD_REDUCE_PROD = 5,    /**< proo */
    HOROVOD_REDUCE_RESERVED     /**< reserved */
} HorovodReduceOp;

const u32 HCCL_MAX_SEGMENT_NUM = 8;   // The max number of gradient segments.

/**
 * @brief the feature of the model
 */
struct model_feature {
    const char *model_name;  /**< The model name */
    u32 gradient_num;        /**< The number of gradients */
    float *gradient_size;    /**< The size of each gradient */
    float *gradient_time;    /**< The BP compution time of each gradient */
};

/**
 * @brief Memory Register Address Struct for Remote Access
 */
struct MemRegisterAddr {
    u64 addr;
    u64 length;
};
/*
 * @brief The max number of memory register address for remote access.
 */
const u32 HCCL_MAX_MEM_REGISTER_NUM = 32;

enum GradSplitForceMode {
    FORCE_NONE,     /**< no force */
    FORCE_SIZE,     /**< force split gradient by size */
    FORCE_RESERVED  /**< reserved */
};

enum OriginalGraphShapeType {
    KNOWN_SHAPE,
    UNKNOWN_SHAPE,
    SHAPE_RESERVED  /**< reserved */
};

enum HcclEventType {
    HCCL_EVENT_SEND_COMPLETION = 0,
    HCCL_EVENT_RECV_REQUEST,
    HCCL_EVENT_RECV_COMPLETION,
    HCCL_EVENT_CONGESTION_RELIEF,
    HCCL_EVENT_RESERVED /**< reserved */
};

const u32 TAG_MAX_LEN = 127; // ����tag ����
using TagAttr = struct TagAttrDef {
    char name[TAG_MAX_LEN + 1]; // tag��ʶ
    // tag��ʶ�Ľ������ݣ��������Ƿ���������ý��սӿڣ�0 = ��, 1 = ��(Ԥ�����ݲ�֧��)��
    // ����activeRecv = 0�������ղ��յ����ݻ��߷�������ʱ������֪ͨ�����ߡ�
    uint32_t activeRecv;
    uint32_t sendCredit; // ���ø�tag����inflight��send����
    uint32_t eventId;
};

using HcclEventMsg = struct HcclEventMsgDef {
    HcclComm comm;
    u32 peerRank;
    u32 tag;
    // 0:HCCL_SEND_COMPLETION; 1:HCCL_RECV_COMPLETION; 2:HCCL_RECV_REQUEST; 3:HCCL_CONGESTION_RELIEF
    u32 hcclEventType;
    union {
        struct {
            u32 reserver;
        } sendCompletionItem;
        struct {
            u32 reserver;
        } recvRequestItem;
        struct {
            u32 reserver;
        } recvCompletionItem;
        struct CongestionReliefItem {
            u32 reserver;
        } congestionReliefItem;
    } desc;
};


/**
* @brief stream handle.
*/
typedef void *rtStream_t;

/**
* @brief model handle.
*/
typedef void *rtModel_t;

struct HcomOperation {
    std::string hcclType;
    void *inputPtr;
    void *outputPtr;
    u64 count;
    HcclDataType dataType;
    HcclReduceOp opType;
    u32 root;

    HcomOperation()
    {
        inputPtr = nullptr;
        outputPtr = nullptr;
        count = 0;
        dataType = HCCL_DATA_TYPE_RESERVED;
        opType = HCCL_REDUCE_RESERVED;
        root = 0;
    }
};

struct HcomRemoteAccessAddrInfo {
    u32 remotetRankID;
    u64 remoteAddr;  // host embedding table address
    u64 localAddr;  // device HBM address
    u64 length;   // Memory Length in Bytes
};

struct HcomAllToAllVParams {
    void *sendbuf;  // device mem
    void *sendcounts;  // device mem;  Type: uint_64
    void *sdispls;  // device mem;  Type: uint_64
    HcclDataType sendtype;
    void *recvbuf;  // device mem
    void *recvcounts;  // device mem;  Type: uint_64
    void *rdispls;  // device mem;  Type: uint_64
    HcclDataType recvtype;
    const char *group;  // not used now
};

struct HcomGatherAllToAllVParams {
    void *addrInfo;  // device mem;  contains host VA[uint_64]:  [addr, length, addr, length, addr, length, ...]
    void *addrInfoCountPerRank;  // device mem;  length: ranksize;  contains addrInfoCounts for every rank
    void *recvbuf;  // device mem
    void *recvcounts;  // device mem;  Type: uint_64
    void *rdispls;  // device mem;  Type: uint_64
    void *gatheredbuf;  // device mem
    s32 addrLength;
    HcclDataType recvtype;
    const char *group;  // not used now
};

typedef enum workMode {
HCCL_MODE_NORMAL = 0, // ��֧���κ�Probe any����֧�־�ȷ��probe
HCCL_MODE_ANY = 1     // ��֧��ANY_SOURCE + ANY_TAG��probe
} WorkMode;

typedef struct tagCommAttr {
    WorkMode mode;  // ͨ�����ڵ�probe����ģʽ
    uint32_t deviceId = 0;
} CommAttr;

typedef void* HcclMessage;
typedef void* HcclRequest;

typedef struct {
    int srcRank;    // ����/̽�⵽��msg/�ŷ�ķ��Ͷ�rank_id��MPI��׼���壬�����߿��Է���
    int tag;        // ����/̽�⵽��msg/�ŷ��tag��MPI��׼���壬�����߿��Է���
    int error;      // ����/̽��Ĵ�����0��no error��others��������̳���MPI��׼���壬�����߿��Է���
    int cancelled;  // ָ��ʵ�֣�����������߷���
    int count;      // ����/̽�⵽��payload��С��ָ��ʵ�֣�����������߷���
} HcclStatus;

#define HCCL_REQUEST_NULL   NULL

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_BASE_H_
