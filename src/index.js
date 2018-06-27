
var array = require("./wish3d.array");
var fs = require("fs");
[].forEach.call(array,function (item,index) {
    var str = ".pipe(replace(/_0x34b6\\\["+index+"\\\]/g,array["+index+"]))\n";
    fs.appendFile("aaa.txt",str)
})