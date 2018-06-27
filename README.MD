说明

安装依赖项

```
yarn
```

执行 gulp 命令

先执行  jsnice 格式化部分代码

然后 执行 replace 命令 ，替换 顶部数组里的变量

写成一个命令会堆栈溢出，分批次执行，一次替换400个变量

```
gulp jsnice
gulp replace_400
gulp replace_800
gulp replace_1200
gulp replace_1600
gulp replace_2000
gulp replace_2400
```

