/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#pragma once
namespace oceanbase { namespace sql {
class ObExecContext;
bool is_ai_column_read_check();
int check_ai_column_strict_read(ObExecContext &context);
}}
