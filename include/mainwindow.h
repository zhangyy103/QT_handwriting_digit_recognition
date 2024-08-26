#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPainter>
#include <QMenu>
#include <QPainterPath>
#include <QMouseEvent>
#include <QDebug>
#include <QTimer>
#include <QTime>
#include <QAction>
#include <QColorDialog>
#include <QInputDialog>
#include <QFileDialog>
#include <QImage>
#include <QProcess>
#include "include/recognition.h"

struct myPoint{
    QPoint point;  //起点坐标
    QPoint move_point;  //移动到的点的坐标

    int p_r;
    int p_g;
    int p_b;
};

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void create_menu(); //创建右键菜单
    void save_canvas_as_image();
public slots:
    void clear_clicked();
    void select_clicked();
    void line_width_clicked();
    void eraser_clicked();
    void submit_clicked();

protected:
    void paintEvent(QPaintEvent* envent) override;
    void mousePressEvent(QMouseEvent* envent) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;


private:
    Ui::MainWindow *ui;

    bool is_clicked = false;
    bool is_eraser = false;

    QPoint mouse_point;
    QPoint mouse_move_point;

    QMenu *mouse_menu = nullptr;
    QFont menu_font;
    QList<myPoint> point_list;

    void erase_lines_in_rect(const QRect &rect);

    QAction *eraser_action = new QAction(tr("使用橡皮擦"), this);

    int lineWidth = 25;  // 默认线条宽度

    int p_R = 0;  // 颜色的红色分量，初始为255（红色）
    int p_G = 0;    // 颜色的绿色分量，初始为0
    int p_B = 0;    // 颜色的蓝色分量，初始为0



};
#endif // MAINWINDOW_H
