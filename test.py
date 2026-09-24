from visionist_client import Visionist          # pip install visionist-client  (~10 s: grpcio/protobuf/zstandard)
import pathlib


#calling CLIP (image + text -> similarity)
b = Visionist("localhost:9061")              # the box's address
img = pathlib.Path("images/clip/test/dog.jpg")
car = pathlib.Path("images/clip/test/car.jpg")

res = b.run(data={"images": [img], "texts": ["a dog","the ocean"]},
            config={"clip": {"command": "process", "parameters": {}}})     # config = the box's section: WHAT to do

print(res)


#calling textemb / sbert (text -> embeddings)
b = Visionist("localhost:9062")

res = b.run(data={"texts": ["a dog", "a car", "grass, sky, bark"]},
            config={"sbert": {"command": "encode", "parameters": {}}})

print(res)


#calling tapnext (frames -> point tracks; stateful, so reset first)
b = Visionist("localhost:9063")
b.reset("tapnext")                     # tapnext accumulates tracks across calls

res = b.run(data={"images": [img, img, img]},
            config={"tapnext": {"command": "track", "parameters": {"grid_size": 30}}})

print(res)


#calling lang_segm (image + prompt -> segmentation)  [slow one: ~1 min on CPU]
b = Visionist("localhost:9064")

res = b.run(data={"images": [img]},
            config={"lang_sam": {"command": "segment", "parameters": {"box_threshold": 0.3, "text_threshold": 0.25},
                                 "text_prompt": ["a dog"]}})

print(res)


#calling opencv (two images -> feature match)
b = Visionist("localhost:9065")

res = b.run(data={"images": [img, car]},
            config={"opencv": {"command": "match", "parameters": {}}})

print(res)
