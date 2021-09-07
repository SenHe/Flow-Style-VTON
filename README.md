# Flow Style: Style-Based Appearance Flow for Virtual Try-On
## Training ( `cd` to the train folder)
### Stage 1: Parser-Based Appearance Flow Style
```
sh scripts/train_PBAFN_stage1_fs.sh
```
### Stage 2: Parser-Based Generator
```
sh scripts/train_PBAFN_e2e_fs.sh
```

### Stage 3: Parser-Free Appearance Flow Style
```
sh scripts/train_PFAFN_stage1_fs.sh
```

### Stage 4: Parser-Free Generator
```
sh scripts/train_PFAFN_e2e_fs.sh
```
