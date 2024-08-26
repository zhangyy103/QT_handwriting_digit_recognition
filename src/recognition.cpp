#include "include/recognition.h"
#include <filesystem>

void python_init(){
    // 设置动态库路径
#if 0
    const char* pythonPath = "../../deeple/Library/bin";
    _putenv_s("PATH", pythonPath);
    // 设置 PYTHONHOME 和 PYTHONPATH
    _putenv_s("PYTHONHOME", "D:/QtProject/hw_digit_recognition/deeple");
    _putenv_s("PYTHONPATH", "D:/QtProject/hw_digit_recognition/deeple/Lib;D:/QtProject/hw_digit_recognition/deeple/DLLs");
#endif
    // 获取当前工作目录
    std::filesystem::path currentPath = std::filesystem::current_path();

    // 构建相对路径
    std::filesystem::path pythonHome = currentPath / "../../deeple";
    std::filesystem::path pythonPath = pythonHome / "Lib;../../deeple/DLLs";
    std::filesystem::path pythonBin = pythonHome / "Library/bin";

    // 设置环境变量
    _putenv_s("PATH", pythonBin.string().c_str());
    _putenv_s("PYTHONHOME", pythonHome.string().c_str());
    _putenv_s("PYTHONPATH", pythonPath.string().c_str());


    // 初始化Python解释器
    Py_Initialize();

    // 判断Python解释器是否已经初始化完成
    if (!Py_IsInitialized()) {
        qDebug() << "Py_Initialize fail";
        return;
    } else {
        qDebug() << "Py_Initialize success";
    }

    return;
}

int recognition(){
    // 初始化 Python 解释器

    // 导入sys模块设置模块地址，以及python脚本路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../../python_scripts')");

    // 加载 python 脚本
    PyObject *pModule = PyImport_ImportModule("recognition");  // 脚本名称，不带.py
    if (!pModule) {
        PyErr_Print();
        return -1;
    }

    // 创建函数指针
    PyObject* pFunc = PyObject_GetAttrString(pModule, "predict_digit");  // 方法名称
    if (!pFunc || !PyCallable_Check(pFunc)) {
        PyErr_Print();
        Py_XDECREF(pModule);
        qDebug() << "python_scripts call fail";
        return -1;
    }

    // 调用函数
    PyObject* pyResult = PyObject_CallObject(pFunc, NULL);
    if (!pyResult) {
        PyErr_Print();
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
        qDebug() << "python_scripts call fail";
        return -1;
    }

    // 获取返回值
    int predict = PyLong_AsLong(pyResult);
    if (PyErr_Occurred()) {
        PyErr_Print();
        Py_XDECREF(pyResult);
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
        qDebug() << "python_scripts call fail";
        return -1;
    }

    qDebug() << "Predicted digit: " << predict;

    // 释放资源
    Py_XDECREF(pyResult);
    Py_XDECREF(pFunc);
    Py_XDECREF(pModule);

    // 终止 Python 解释器
    return predict;
}
