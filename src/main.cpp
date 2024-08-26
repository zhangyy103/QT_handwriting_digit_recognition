#include "include/mainwindow.h"
#include <QApplication>
#include "include/recognition.h"

int main(int argc, char *argv[])
{
#if 1
    python_init();
#endif
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
    Py_Finalize();
}
