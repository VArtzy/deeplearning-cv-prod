Deployment

Compile the model

Ways:

prototype (UI + model, eg: gradio)
web server (inside)
model service (REST, etc) // most recommended
edge (model in client)

Use CPU for inference is good unless you need big performance and find problem (use GPU then, but make a batch/paralel can help)
