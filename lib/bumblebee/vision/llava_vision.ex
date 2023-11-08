defmodule Bumblebee.Vision.LlavaVision do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 50282,
        doc: """
        the vocabulary size of the Mpt model. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      model_size: [
        default: 4096,
        doc: "the dimensionality of the embeddings and hidden states."
      ],
      mm_hidden_size: [
        default: 1024,
        doc: "the size of the hidden representations for vision model."
      ],
      image_size: [
        default: 336,
        doc: "the size of the input spatial dimensions"
      ],
      patch_size: [
        default: 14,
        doc: "the size of the patch spatial dimensions"
      ],
      hidden_size: [
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 4096,
        doc: "the dimensionality of intermediate layers"
      ],
      num_blocks: [
        default: 24,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_channels: [
        default: 3,
        doc: "the number of channels in the input"
      ],
      projection_size: [
        default: 768,
        doc: "the dimensionality of the projection layers"
      ],
      projection_hidden_size: [
        default: 4096,
        doc: "the dimensionality of the projection hidden layers"
      ],
      activation: [
        default: :quick_gelu,
        doc: "the activation function"
      ],
      attention_dropout_rate: [
        default: 0.0,
        doc: "the dropout rate for attention weights"
      ],
      layer_norm_epsilon: [
        default: 1.0e-05,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      initializer_factor: [
        default: 1.0,
        doc: "the factor of the normal initializer used for initializing kernel parameters"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions
      ])

  @moduledoc """
  The LLaVA vision MPT model.

  ## Architectures

    * `:base` - the base image model

  ## Inputs

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          model_size: {"d_model", number()},
          mm_hidden_size: {"mm_hidden_size", number()},
          image_size: {"image_size", number()},
          patch_size: {"patch_size", number()},
          hidden_size: {"hidden_size", number()},
          intermediate_size: {"intermediate_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_channels: {"num_channels", number()},
          projection_size: {"projection_dim", number()},
          projection_hidden_size: {"proj_hidden_size", number()},
          activation: {"hidden_act", atom()},
          attention_dropout_rate: {"attention_dropdout", number()},
          layer_norm_epsilon: {"layer_norm_eps", number()},
          initializer_scale: {"initializer_range", number()},
          initializer_factor: {"initializer_factor", number()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end
end
