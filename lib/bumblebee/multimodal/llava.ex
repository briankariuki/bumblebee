defmodule Bumblebee.Multimodal.Llava do
  alias Bumblebee.Shared

  options =
    [
      [
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
end
