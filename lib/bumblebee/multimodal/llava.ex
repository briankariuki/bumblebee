defmodule Bumblebee.Multimodal.Llava do
  alias Bumblebee.Shared

  options =
    [
      [
        text_spec: [
          default: nil,
          doc: "the specification of the text model. See `Bumblebee.Text.BlipText` for details"
        ],
        vision_spec: [
          default: nil,
          doc:
            "the specification of the vision model. See `Bumblebee.Vision.BlipVision` for details"
        ],
        vocab_size: [
          default: 32000,
          doc: """
          the vocabulary size of the token embedding. This corresponds to the number of distinct
          tokens that can be represented in model input and output
          """
        ],
        max_positions: [
          default: 4096,
          doc: """
          the vocabulary size of the position embedding. This corresponds to the maximum sequence
          length that this model can process.
          """
        ],
        hidden_size: [
          default: 4096,
          doc: "the dimensionality of hidden layers"
        ],
        intermediate_size: [
          default: 11008,
          doc: "the dimensionality of intermediate layers"
        ],
        num_blocks: [
          default: 32,
          doc: "the number of Transformer blocks in the model"
        ],
        num_attention_heads: [
          default: 32,
          doc: "the number of attention heads for each attention layer in the model"
        ],
        num_key_value_heads: [
          default: nil,
          doc: "the number of key value heads for each attention layer in the model"
        ],
        activation: [
          default: :silu,
          doc: "the activation function"
        ],
        layer_norm_epsilon: [
          default: 1.0e-05,
          doc: "the epsilon used by RMS normalization layers"
        ],
        initializer_scale: [
          default: 0.02,
          doc:
            "the standard deviation of the normal initializer used for initializing kernel parameters"
        ]
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions,
        :num_labels,
        :id_to_label
      ]) ++ Shared.token_options(pad_token_id: 0)

  @moduledoc """
  LLaVA model family

  ## Architectures
    * `:base` - plain LLaVA without any head on top

    * `:for_causal_language_modeling` - LLaVA with a language modeling
      head. The head returns logits for each token in the original
      sequence

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

        Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", atom()},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{}
    end
  end
end
