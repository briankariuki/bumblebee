defmodule Bumblebee.Text.LlavaTokenizer do
  @moduledoc """
  Llama tokenizer.
  """

  import Bumblebee.Shared

  tokenizer_impl(
    special_tokens: %{
      eos: "</s>",
      unk: "<unk>",
      sep: "</s>",
      # Llama doesn't originally have a pad token, however when necessary
      # we pad with the EOS token
      pad: "</s>"
    },
    additional_special_tokens: [
      "<image>",
      "<im_patch>",
      "<im_start>",
      "<im_end>",
      "<image-placeholder>"
    ]
  )
end
