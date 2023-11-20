defmodule Bumblebee.Multimodal.Llava do
  alias Bumblebee.Shared

  options =
    [
      text_spec: [
        default: nil,
        doc: "the specification of the text model. See `Bumblebee.Text.LlavaText` for details"
      ],
      vision_spec: [
        default: nil,
        doc:
          "the specification of the vision model. See `Bumblebee.Vision.LlavaVision` for details"
      ]
    ] ++
      Shared.common_options([
        :output_hidden_states,
        :output_attentions
      ])

  @moduledoc """
  LLaVA model family

  ## Architectures
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

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"pixel_values"` - `{batch_size, image_size, image_size, num_channels}`

      Featurized image pixel values.

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

  defstruct [architecture: :for_causal_language_modelling] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:for_causal_language_modelling]

  @impl true
  def config(spec, opts \\ []) do
    Shared.put_config_attrs(spec, opts)
  end

  @impl true
  def input_template(%{vision_spec: vision_spec}) do
    vision_shape = {1, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    %{
      "input_ids" => Nx.template({1, 1}, :u32),
      "pixel_values" => Nx.template(vision_shape, :f32)
    }
  end

  @impl true
  def model(%__MODULE__{architecture: :for_causal_language_modelling} = spec) do
    %{text_spec: text_spec, vision_spec: vision_spec} = spec

    text_shape = {nil, nil}
    vision_shape = {nil, vision_spec.image_size, vision_spec.image_size, vision_spec.num_channels}

    hidden_shape = {nil, nil, text_spec.hidden_size}
    attention_head_mask_shape = {text_spec.num_blocks, text_spec.num_attention_heads}

    inputs =
      Bumblebee.Utils.Model.inputs_to_map([
        Axon.input("input_ids", shape: text_shape),
        Axon.input("attention_mask", optional: true, shape: text_shape),
        Axon.input("position_ids", optional: true, shape: text_shape),
        Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
        Axon.input("input_embeddings", optional: true, shape: hidden_shape),
        Axon.input("pixel_values", shape: vision_shape),
        Axon.input("cache", optional: true)
      ])

    vision_model =
      vision_spec
      |> Bumblebee.configure(
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_hidden_states
      )
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("vision_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "pixel_values" => inputs["pixel_values"]
      })

    image_features =
      vision_model
      |> Axon.nx(& &1.hidden_states)
      |> projector(vision_spec)

    # TODO: Determine text model inputs

    text_model =
      text_spec
      |> Bumblebee.configure(
        output_hidden_states: spec.output_hidden_states,
        output_attentions: spec.output_hidden_states,
        architecture: :for_causal_language_modelling
      )
      |> Bumblebee.build_model()
      |> Bumblebee.Utils.Axon.prefix_names("text_model.")
      |> Bumblebee.Utils.Axon.plug_inputs(%{
        "input_ids" => inputs["input_ids"],
        "input_embeddings" => inputs["input_embeddings"],
        "attention_mask" => inputs["attention_mask"],
        "attention_head_mask" => inputs["attention_head_mask"],
        "position_ids" => inputs["position_ids"],
        "cache" => inputs["cache"]
      })

    # TODO: Determine model outputs
    Layers.output(%{})
  end

  defp projector(image_embeddings, vision_spec) do
    image_embeddings[-2][1..-1//1]
    |> Axon.dense(vision_spec.projection_hidden_size, use_bias: false)
    |> Axon.gelu()
    |> Axon.dense(vision_spec.projection_hidden_size, use_bias: false)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      {text_data, data} = Map.pop(data, "text_config", %{})
      {vision_data, data} = Map.pop(data, "vision_config", %{})

      text_spec =
        Bumblebee.Text.LlavaText
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(text_data)

      vision_spec =
        Bumblebee.Vision.LlavaVision
        |> Bumblebee.configure()
        |> Bumblebee.HuggingFace.Transformers.Config.load(vision_data)

      opts =
        Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts ++ [text_spec: text_spec, vision_spec: vision_spec])
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    alias Bumblebee.HuggingFace.Transformers

    def params_mapping(spec) do
      text_mapping =
        spec.text_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("text_model", nil)

      vision_mapping =
        spec.vision_spec
        |> Transformers.Model.params_mapping()
        |> Transformers.Utils.prefix_params_mapping("vision_model", nil)

      %{
        "text_projection" => "text_projection",
        "visual_projection" => "visual_projection"
      }
      |> Map.merge(text_mapping)
      |> Map.merge(vision_mapping)
    end
  end
end
