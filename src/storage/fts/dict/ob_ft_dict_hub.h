/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_HUB_H_
#define _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_HUB_H_

#include "lib/allocator/page_arena.h"
#include "lib/charset/ob_charset.h"
#include "lib/lock/ob_bucket_lock.h"
#include "storage/fts/dict/ob_ft_dict_def.h"

namespace oceanbase
{
namespace storage
{
// typedef uint64_t ObFTTableID;
// typedef ObFTTableID ObFTDictTableID;

class ObFTDictInfo
{
public:
  ObFTDictInfo()
      : name_(""),
        type_(ObFTDictType::DICT_TYPE_INVALID),
        charset_(CHARSET_INVALID),
        version_(0),
        range_count_(0)
  {
  }

public:
  char name_[2048]; // for now
  ObFTDictType type_;
  ObCharsetType charset_;
  int64_t version_; // in memory
  int32_t range_count_;
};

struct ObFTDictInfoKey
{
public:
  ObFTDictInfoKey()
      : type_(static_cast<uint64_t>(ObFTDictType::DICT_TYPE_INVALID))
  {
  } // default constructor
  ObFTDictInfoKey(const uint64_t type)
      : type_(type)
  {
  }
  int hash(uint64_t &hash_value) const
  {
    int ret = OB_SUCCESS;
    hash_value = hash();
    return ret;
  }

  uint64_t hash() const
  {
    uint64_t hash = 0;
    hash = common::murmurhash(&type_, sizeof(int64_t), hash);
    return hash;
  }

  bool operator==(const ObFTDictInfoKey &other) const
  {
    return type_ == other.type_;
  }

  int compare(const ObFTDictInfoKey &other) const
  {
    int ret = 0;
    if (0 == ret) {
      ret = type_ - other.type_;
    }
    return ret;
  }

private:
  uint64_t type_;
  // name
};

class ObFTCacheRangeContainer;
class ObIFTDict;
class ObFTDictHub
{
public:
  ObFTDictHub()
      : is_inited_(false),
        dict_map_(),
        rw_dict_lock_(),
        shared_dict_alloc_(lib::ObMemAttr("IK Shared Dict"))
  {
  }
  ~ObFTDictHub() {}

  int init();

  int destroy();

  int build_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container);

  int load_cache(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container);

  /**
   * Get the shared dict of the descriptor, which is built on first use and then
   * reused by all tokenizer instances. It avoids loading the dict ranges and
   * rebuilding the range dict for every document.
   */
  int get_shared_dict(const ObFTDictDesc &desc, ObIFTDict *&dict);

private:
  int get_dict_info(const ObFTDictInfoKey &key, ObFTDictInfo &info);

  int put_dict_info(const ObFTDictInfoKey &key, const ObFTDictInfo &info);

  int load_cache_inner(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container);

  int build_cache_inner(const ObFTDictDesc &desc, ObFTCacheRangeContainer &container);

  int build_shared_dict(const ObFTDictDesc &desc, const int64_t idx, ObIFTDict *&dict);

private:
  static constexpr int64_t MAX_SHARED_DICT_NUM = 4; // DICT_TYPE_INVALID ~ DICT_IK_STOP
  struct ObFTSharedDict final
  {
    ObFTCacheRangeContainer *container_ = nullptr;
    ObIFTDict *dict_ = nullptr;
  };

private:
  bool is_inited_;
  // holds info of dict
  hash::ObHashMap<ObFTDictInfoKey, ObFTDictInfo> dict_map_;
  ObBucketLock rw_dict_lock_;
  // pin the dict cache ranges once, so that shared dicts stay valid
  ObFTSharedDict shared_dicts_[MAX_SHARED_DICT_NUM];
  ObArenaAllocator shared_dict_alloc_;
};

} //  namespace storage
} //  namespace oceanbase

#endif // _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_HUB_H_
