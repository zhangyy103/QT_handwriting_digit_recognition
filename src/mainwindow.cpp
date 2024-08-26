#include "include/mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    create_menu();

    this->setWindowTitle("手写数字识别");

    menu_font.setPointSize(5);
    menu_font.setFamily("Microsoft YaHei");

    this->setFixedSize(1280, 720);

    ui->label->setText(QString("手写数字识别系统"));

    this->setWindowIcon(QIcon(":/yuzu.ico"));


}

MainWindow::~MainWindow()
{
    delete ui;
}

//创建右键菜单
void MainWindow::create_menu(){
    mouse_menu = new QMenu(this);
    QAction *clear_action = new QAction(tr("清除画布"), this);
    QAction *select_action = new QAction(tr("选择颜色"), this);
    QAction *lineWidth_action = new QAction(tr("选择线宽"), this);
    //QAction *eraser_action = new QAction(tr("使用橡皮擦"), this);

    mouse_menu->addAction(select_action);  // 将"选择颜色"动作添加到菜单
    mouse_menu->addAction(lineWidth_action);  // 将"线条宽度"动作添加到菜单
    mouse_menu->addAction(eraser_action);
    mouse_menu->addSeparator();  // 添加分隔符
    mouse_menu->addAction(clear_action);  // 将"清除"动作添加到菜单

    connect(clear_action, &QAction::triggered, this, &MainWindow::clear_clicked);
    connect(select_action, &QAction::triggered, this, &MainWindow::select_clicked);
    connect(lineWidth_action, &QAction::triggered, this, &MainWindow::line_width_clicked);
    connect(eraser_action, &QAction::triggered, this, &MainWindow::eraser_clicked);
    connect(ui -> btn_submit, &QPushButton::clicked, this, &MainWindow::submit_clicked);
}

void MainWindow::paintEvent(QPaintEvent *)
{
    QPainter painter(this);

    // 绘制窗口背景
    painter.fillRect(rect(), Qt::white);  // 设置窗口的背景色为白色

    // 绘制超出 x 轴大于 720 的部分的背景色
    QRect outsideRect(720, 0, width() - 720, height());
    painter.fillRect(outsideRect, Qt::lightGray);  // 设置背景色为浅灰色

    painter.setRenderHint(QPainter::Antialiasing);  // 启用抗锯齿

    QRect drawingArea(0, 0, 720, 720);  // 限制绘制区域
    painter.setClipRect(drawingArea);  // 只允许在此区域内绘制

    QPainterPath path;
    path.setFillRule(Qt::WindingFill);  // 设置填充规则
    for (const myPoint &mypoint : point_list) {
        path.moveTo(mypoint.point);
        path.lineTo(mypoint.move_point);
    }

    QPen pen(QColor(p_R, p_G, p_B), lineWidth);
    painter.setPen(pen);
    painter.drawPath(path);
}



void MainWindow::mousePressEvent(QMouseEvent *event)
{
    if (event->buttons() == Qt::LeftButton)  // 判断是否是鼠标左键按下
    {
        is_clicked = true;  // 设置左键按下标志
        mouse_point = event->pos();  // 记录按下点的位置
    }
}
#if 0
void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() == Qt::LeftButton && is_clicked)  // 如果鼠标左键按下且移动
    {
        mouse_move_point = event->pos();  // 记录移动点的位置
        myPoint mypoint;
        mypoint.point = mouse_point;  // 设置起点位置
        mypoint.move_point = mouse_move_point;  // 设置终点位置
        mypoint.p_r = p_R;  // 设置颜色的红色分量
        mypoint.p_g = p_G;  // 设置颜色的绿色分量
        mypoint.p_b = p_B;  // 设置颜色的蓝色分量

        point_list.append(mypoint);  // 将点添加到列表

        mouse_point = mouse_move_point;  // 更新起点为当前点的位置
    }
    update();  // 触发重绘事件，调用paintEvent
}
#endif
void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() == Qt::LeftButton && is_clicked)
    {
        if (is_eraser)
        {
            // 橡皮擦模式
            QPoint current_point = event->pos();
            QRect eraserRect(current_point - QPoint(lineWidth / 2, lineWidth / 2),
                            QSize(lineWidth*1.5, lineWidth*1.5));

            // 更新擦除区域
            erase_lines_in_rect(eraserRect);
        }
        else
        {
            // 绘制模式
            mouse_move_point = event->pos();

            myPoint mypoint;
            mypoint.point = mouse_point;
            mypoint.move_point = mouse_move_point;
            mypoint.p_r = p_R;
            mypoint.p_g = p_G;
            mypoint.p_b = p_B;

            point_list.append(mypoint);

            QRect updateRect(mouse_point, mouse_move_point);
            updateRect = updateRect.normalized();
            updateRect.adjust(-lineWidth, -lineWidth, lineWidth, lineWidth);

            update(updateRect);

            mouse_point = mouse_move_point;
        }
    }
}




void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    is_clicked = false;  // 重置左键按下标志

    if (event->button() == Qt::RightButton)  // 判断是否是鼠标右键释放
    {
        mouse_menu->move(mapToGlobal(event->pos()));  // 移动菜单到鼠标点击的位置
        mouse_menu->show();  // 显示右键菜单
    }
}

// 槽函数：清除所有绘制的内容
void MainWindow::clear_clicked()
{
    point_list.clear();  // 清空点列表
    ui->label->setText(QString("手写数字识别系统"));
    update();  // 触发重绘事件，调用paintEvent
}

void MainWindow::select_clicked()
{
    QColor color = QColorDialog::getColor(QColor(0, 0, 0));  // 打开颜色选择对话框，初始颜色为黑色

    p_R = color.red();    // 获取选择颜色的红色分量
    p_G = color.green();  // 获取选择颜色的绿色分量
    p_B = color.blue();   // 获取选择颜色的蓝色分量
}

void MainWindow::line_width_clicked()
{
    bool ok;
    int newLineWidth = QInputDialog::getInt(this, tr("Line Width"),
                        tr("Select line width:"), lineWidth, 1, 100, 1, &ok);
    if (ok) {
        lineWidth = newLineWidth;  // 更新线条宽度
    }
}

void MainWindow::erase_lines_in_rect(const QRect &rect)
{
    QList<myPoint> newList;

    for (const myPoint &line : point_list)
    {
        // 计算线段的bounding box
        QRect lineRect(QRect(line.point, line.move_point).normalized());

        // 判断线段是否与橡皮擦区域相交
        if (!lineRect.intersects(rect))
        {
            newList.append(line);
        }
    }

    point_list = newList;
    update();  // 更新视图
}

void MainWindow::eraser_clicked()
{
    is_eraser = !is_eraser;  // 切换橡皮擦状态
    if (is_eraser) {
        eraser_action->setText(tr("使用画笔"));  // 更新菜单项文本
        this->setCursor(Qt::CrossCursor);
    } else {
        eraser_action->setText(tr("使用橡皮擦"));
        this->setCursor(Qt::ArrowCursor);
    }
}

void MainWindow::save_canvas_as_image(){
    QPixmap pixmap(720, 720);  // 创建一个 QPixmap 对象，大小为画布的大小
    QPainter painter(&pixmap);
    painter.fillRect(pixmap.rect(), Qt::white);  // 设置背景为白色

    // 绘制内容到 QPixmap 上
    QPainterPath path;
    path.setFillRule(Qt::WindingFill);
    for (const myPoint &mypoint : point_list) {
        path.moveTo(mypoint.point);
        path.lineTo(mypoint.move_point);
    }

    QPen pen(QColor(p_R, p_G, p_B), lineWidth);
    painter.setPen(pen);
    painter.drawPath(path);

    // 保存 QPixmap 为图片文件
    //QString filePath = QFileDialog::getSaveFileName(this, tr("Save Image"), "", tr("PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"));
    //if (!filePath.isEmpty()) {
    //    pixmap.save(filePath);
    //}
    // 自动保存图片到指定路径和格式
    QString filePath = "../../assets/saved_canvas.png";
    pixmap.save(filePath, "PNG");
}

void MainWindow::submit_clicked(){
    save_canvas_as_image();
    ui->label->setText(QString("识别中 . . ."));
    QApplication::processEvents();
    int rec = recognition();
    ui->label->setText(QString("识别值: %1").arg(rec));
}
