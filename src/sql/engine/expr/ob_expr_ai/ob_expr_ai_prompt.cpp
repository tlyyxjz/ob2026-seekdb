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

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_prompt.h"
#include <regex>
#include "ob_ai_func.h"
#include "lib/utility/utility.h"
#include "common/json_type/ob_json_common.h"
#include "common/json_type/ob_json_base.h"


using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{

ObExprAIPrompt::ObExprAIPrompt(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                    T_FUN_SYS_AI_PROMPT, 
                    N_AI_PROMPT, 
                    MORE_THAN_ZERO,
                    VALID_FOR_GENERATED_COL, 
                    NOT_ROW_DIMENSION) 
{
}

ObExprAIPrompt::~ObExprAIPrompt() 
{
}

int ObExprAIPrompt::calc_result_typeN(ObExprResType &type,
                                    ObExprResType *types_stack,
                                    int64_t param_num,
                                    common::ObExprTypeCtx &type_ctx) const 
{
  int ret = OB_SUCCESS;
  bool is_null_res = false;
  ObExprResType template_type = types_stack[0];
  if (!ob_is_string_tc(template_type.get_type())) {
    ret = OB_ERR_INVALID_TYPE_FOR_OP;
  } else {
    types_stack[0].set_calc_type(ObVarcharType);
    types_stack[0].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
  }

  for (int64_t i = 1; i < param_num && OB_SUCC(ret); i++) {
    if (ob_is_string_tc(types_stack[i].get_type())) {
      types_stack[i].set_calc_type(ObVarcharType);
      types_stack[i].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    } else if (ob_is_json(types_stack[i].get_type())) {
      // JSON arguments keep their structure in the prompt args array, so the
      // argument is evaluated as JSON rather than being converted to a string.
      types_stack[i].set_calc_type(ObJsonType);
    } else {
      ret = OB_ERR_INVALID_TYPE_FOR_OP;
    }
  }

  if (OB_SUCC(ret)) {
    type.set_json();
    type.set_length((ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType]).get_length());
  }
  return ret;
}

int ObExprAIPrompt::eval_ai_prompt(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *template_datum = NULL;
  ObExpr *arg0 = expr.args_[0];
  ObString template_str;
  ObJsonArray *args_array = NULL;
  ObJsonObject *prompt_object = NULL;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  
  MultimodeAlloctor tmp_allocator(tmp_alloc_g.get_allocator());
  lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_PROMPT));

  if (OB_FAIL(tmp_allocator.eval_arg(arg0, ctx, template_datum))) {
  } else if (template_datum->is_null()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_,
                                                               tmp_allocator,
                                                               *template_datum,
                                                               arg0->datum_meta_,
                                                               arg0->obj_meta_.has_lob_header(),
                                                               template_str))) {
    // The template may come from a TEXT / LONGTEXT column, whose datum carries a lob
    // header and points at out of line storage, so the text has to be read out first.
  }

  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(tmp_allocator, args_array))) {
  } else {
    for (int64_t i = 1; i < expr.arg_cnt_ && OB_SUCC(ret); i++) {
      ObDatum *datum = NULL;
      ObExpr *arg = expr.args_[i];
      ObString arg_str;
      ObJsonString *arg_json_str = NULL;
      if (OB_FAIL(tmp_allocator.eval_arg(arg, ctx, datum))) {
      } else if (datum->is_null()) {
        // A NULL argument is a type error for ai_prompt regardless of the declared
        // type of the argument, so the NULL check must come before type dispatch.
        ret = OB_ERR_INVALID_TYPE_FOR_OP;
      } else if (ob_is_string_tc(arg->datum_meta_.type_)) {
        if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_,
                                                              tmp_allocator,
                                                              *datum,
                                                              arg->datum_meta_,
                                                              arg->obj_meta_.has_lob_header(),
                                                              arg_str))) {
        } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(tmp_allocator, arg_str, arg_json_str))) {
        } else if (OB_FAIL(args_array->append(arg_json_str))) {
        }
      } else if (ob_is_json(arg->datum_meta_.type_)) {
        // JSON arguments (JSON array, JSON object, JSON scalar) are appended with
        // their structure preserved. The value is reparsed from the datum instead of
        // being taken from the expression's json tree, because an expression
        // argument's per-row json tree is not guaranteed to be materialised.
        ObIJsonBase *json_base = NULL;
        arg_str = datum->get_string();
        if (arg_str.empty()) {
          ret = OB_ERR_INVALID_TYPE_FOR_OP;
        } else if (OB_FAIL(ObJsonBaseFactory::get_json_base(&tmp_allocator, arg_str,
                                                            ObJsonInType::JSON_TREE,
                                                            ObJsonInType::JSON_TREE,
                                                            json_base))) {
          LOG_WARN("fail to parse json argument of ai_prompt", K(ret));
        } else if (OB_ISNULL(json_base)) {
          ret = OB_ERR_UNEXPECTED;
        } else if (OB_FAIL(args_array->append(json_base))) {
        }
      } else {
        ret = OB_ERR_INVALID_TYPE_FOR_OP;
      }
    }
  }

  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObAIFuncPromptObjectUtils::construct_prompt_object(tmp_allocator, template_str, args_array, prompt_object))) {
  } else {
    ObString raw_str;
    if (OB_FAIL(prompt_object->get_raw_binary(raw_str, &tmp_allocator))) {
    } else if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, res, raw_str))) {
    }
  }
  return ret;
}



int ObExprAIPrompt::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                        const ObRawExpr &raw_expr,
                        ObExpr &rt_expr) const 
{
  UNUSED(expr_cg_ctx);
  UNUSED(raw_expr);
  int ret = OB_SUCCESS;
  rt_expr.eval_func_ = ObExprAIPrompt::eval_ai_prompt;
  return ret;
}

} // namespace sql
} // namespace oceanbase
