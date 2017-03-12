#!/bin/bash
th train.lua -data data/demo-train.t7 -save_model model -gpuid '1' -adaptive_softmax '{2000,10000}'
