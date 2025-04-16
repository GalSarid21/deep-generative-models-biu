#!/bin/sh

# relevant args to set default value
DEFAULTS="
  experiment gold-idx-change
  num_docs 10
  model tiiuae/Falcon3-Mamba-7B-Instruct
"

PASSED_ARGS="$@"
FINAL_ARGS="$PASSED_ARGS"

# check if an argument (e.g., 'model') was passed
arg_passed() {
  echo "$PASSED_ARGS" | grep -Eq -- "--$1(=| |$)"
}

# loop over defaults and append only if not passed
for pair in $DEFAULTS; do
  if [ -z "$key" ]; then
    key=$pair
  else
    value=$pair
    if ! arg_passed "$key"; then
      FINAL_ARGS="$FINAL_ARGS --$key $value"
    fi
    key=""
  fi
done

exec python /app/main.py $FINAL_ARGS