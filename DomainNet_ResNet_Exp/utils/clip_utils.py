def clip_config(model_cfg, state_dict, pretrained=True):

    if pretrained:
        counts: list = [
            len(
                {
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                }
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32
        embed_dim = state_dict["text_projection"].shape[1]
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"module.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_heads = state_dict["module.layer1.0.conv1.weight"].shape[0] * 32 // 64
        vision_width = state_dict["module.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["module.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        assert (
            output_width**2 + 1
            == state_dict["module.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32
        embed_dim = state_dict["module.attnpool.c_proj.bias"].shape[0]

    model_cfg["layers"] = vision_layers
    model_cfg["heads"] = vision_heads
    model_cfg["input_resolution"] = image_resolution
    model_cfg["output_dim"] = embed_dim
    model_cfg["width"] = vision_width
