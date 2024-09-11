import sys
import os
import sqlite3
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QTableView, QComboBox, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
                             QMessageBox)
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.animation import FuncAnimation
import pandas as pd

# Optional imports
try:
    import magic
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

# Enable high DPI scaling for better visuals
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class DatabaseAnalysisThread(QThread):
    update_progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path

    def run(self):
        try:
            results = {}
            self.update_progress.emit("Analyzing database structure...")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            results['tables'] = [table[0] for table in tables]

            table_info = {}
            for table in results['tables']:
                self.update_progress.emit(f"Analyzing table: {table}")
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                table_info[table] = {
                    'columns': [col[1] for col in columns],
                    'row_count': cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                }
            results['table_info'] = table_info

            conn.close()
            self.finished.emit(results)
        except Exception as e:
            self.update_progress.emit(f"Error in database analysis: {str(e)}")
            self.update_progress.emit(traceback.format_exc())
            self.finished.emit({'error': str(e)})

class ModelAnalysisThread(QThread):
    update_progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            results = {}
            self.update_progress.emit("Inspecting file...")
            
            file_size = os.path.getsize(self.model_path)
            file_type = magic.from_file(self.model_path) if magic else "Unknown"
            file_extension = os.path.splitext(self.model_path)[1].lower()
            
            results['file_info'] = f"File size: {file_size} bytes\nFile type: {file_type}\nFile extension: {file_extension}"
            
            self.update_progress.emit("Attempting to load model...")
            
            if torch:
                try:
                    state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
                    results['framework'] = 'PyTorch'
                    
                    if isinstance(state_dict, dict):
                        results['structure'] = self.analyze_dict_structure(state_dict)
                        results['param_count'] = self.count_parameters(state_dict)
                    else:
                        results['structure'] = str(state_dict)
                        results['param_count'] = sum(p.numel() for p in state_dict.parameters()) if hasattr(state_dict, 'parameters') else 0
                except Exception as e:
                    self.update_progress.emit(f"PyTorch loading failed: {str(e)}")
            
            if tf and 'framework' not in results:
                try:
                    model = tf.keras.models.load_model(self.model_path)
                    results['framework'] = 'TensorFlow'
                    results['structure'] = str(model.summary())
                    results['param_count'] = model.count_params()
                except Exception as e:
                    self.update_progress.emit(f"TensorFlow loading failed: {str(e)}")
            
            if 'framework' not in results:
                raise ValueError("Failed to load model with both PyTorch and TensorFlow.")
            
            self.update_progress.emit("Model analyzed successfully.")
            self.finished.emit(results)
        except Exception as e:
            self.update_progress.emit(f"Error in model analysis: {str(e)}")
            self.update_progress.emit(traceback.format_exc())
            results['error'] = str(e)
            self.finished.emit(results)

    def analyze_dict_structure(self, d, indent=0):
        structure = ""
        for key, value in d.items():
            structure += "  " * indent + str(key) + ":\n"
            if isinstance(value, dict):
                structure += self.analyze_dict_structure(value, indent + 1)
            elif isinstance(value, torch.Tensor):
                structure += "  " * (indent + 1) + f"Tensor shape: {value.shape}\n"
            elif hasattr(value, '__dict__'):
                structure += self.analyze_dict_structure(value.__dict__, indent + 1)
            else:
                structure += "  " * (indent + 1) + f"Type: {type(value)}\n"
        return structure

    def count_parameters(self, d):
        count = 0
        if isinstance(d, dict):
            for value in d.values():
                if isinstance(value, torch.Tensor):
                    count += value.numel()
                elif isinstance(value, dict):
                    count += self.count_parameters(value)
                elif hasattr(value, 'parameters'):
                    count += sum(p.numel() for p in value.parameters())
        elif hasattr(d, 'parameters'):
            count = sum(p.numel() for p in d.parameters())
        return count

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Database and Model Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_db_button = QPushButton("Load Database")
        self.load_db_button.clicked.connect(self.load_database)
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_files_button = QPushButton("Load Files")
        self.load_files_button.clicked.connect(self.load_files)
        button_layout.addWidget(self.load_db_button)
        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.load_files_button)
        self.layout.addLayout(button_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        # Database tab
        self.db_tab = QWidget()
        self.db_layout = QVBoxLayout(self.db_tab)
        self.table_selector = QComboBox()
        self.table_selector.currentTextChanged.connect(self.load_table_data)
        self.db_layout.addWidget(self.table_selector)
        self.table_view = QTableView()
        self.db_layout.addWidget(self.table_view)
        self.tab_widget.addTab(self.db_tab, "Database")

        # Model tab
        self.model_tab = QWidget()
        self.model_layout = QVBoxLayout(self.model_tab)
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_layout.addWidget(self.model_info)
        
        # Graph widget
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.model_layout.addWidget(self.canvas)
        
        # Table widget
        self.param_table = QTableWidget()
        self.model_layout.addWidget(self.param_table)
        
        self.tab_widget.addTab(self.model_tab, "Model Info")

        # Log widget
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.layout.addWidget(self.log_widget)

        self.db = QSqlDatabase.addDatabase("QSQLITE")

    def load_database(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Database", "", "SQLite Database Files (*.db *.sqlite);;All Files (*)")
            if file_name:
                if os.path.exists(file_name):
                    response = QMessageBox.question(self, "Database Exists", "This database already exists. Do you want to update the information?",
                                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if response == QMessageBox.No:
                        return
                self.db.setDatabaseName(file_name)
                if self.db.open():
                    self.log("Database opened successfully.")
                    self.analyze_database(file_name)
                else:
                    self.log("Failed to open database.")
        except Exception as e:
            self.log(f"Error loading database: {str(e)}")
            self.log(traceback.format_exc())

    def load_files(self):
        try:
            file_names, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "CSV Files (*.csv);;All Files (*)")
            if file_names:
                for file_name in file_names:
                    self.import_csv_to_db(file_name)
        except Exception as e:
            self.log(f"Error loading files: {str(e)}")
            self.log(traceback.format_exc())

    def import_csv_to_db(self, file_path):
        try:
            df = pd.read_csv(file_path)
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            
            conn = sqlite3.connect(self.db.databaseName())
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            self.log(f"Imported {file_path} into database as table {table_name}.")
        except Exception as e:
            self.log(f"Error importing CSV to database: {str(e)}")
            self.log(traceback.format_exc())

    def analyze_database(self, db_path):
        try:
            self.db_thread = DatabaseAnalysisThread(db_path)
            self.db_thread.update_progress.connect(self.log)
            self.db_thread.finished.connect(self.update_database_info)
            self.db_thread.start()
        except Exception as e:
            self.log(f"Error starting database analysis: {str(e)}")
            self.log(traceback.format_exc())

    def update_database_info(self, results):
        try:
            if 'error' in results:
                self.log(f"Error in database analysis: {results['error']}")
                return

            self.table_selector.clear()
            self.table_selector.addItems(results['tables'])
            
            info = "Database Structure:\n\n"
            for table, data in results['table_info'].items():
                info += f"Table: {table}\n"
                info += f"Columns: {', '.join(data['columns'])}\n"
                info += f"Row count: {data['row_count']}\n\n"
            
            self.model_info.setText(info)
            self.log("Database analysis completed.")
        except Exception as e:
            self.log(f"Error updating database info: {str(e)}")
            self.log(traceback.format_exc())

    def load_table_data(self, table_name):
        try:
            model = QSqlTableModel(self, self.db)
            model.setTable(table_name)
            model.select()
            self.table_view.setModel(model)
        except Exception as e:
            self.log(f"Error loading table data: {str(e)}")
            self.log(traceback.format_exc())

    def load_model(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "Model Files (*.pth *.h5);;All Files (*)")
            if file_name:
                self.analyze_model(file_name)
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            self.log(traceback.format_exc())

    def analyze_model(self, model_path):
        try:
            self.model_thread = ModelAnalysisThread(model_path)
            self.model_thread.update_progress.connect(self.log)
            self.model_thread.finished.connect(self.update_model_info)
            self.model_thread.start()
        except Exception as e:
            self.log(f"Error starting model analysis: {str(e)}")
            self.log(traceback.format_exc())

    def update_model_info(self, results):
        try:
            info = f"File Information:\n{results.get('file_info', 'N/A')}\n\n"
            
            if 'error' in results:
                info += f"Error: {results['error']}\n"
            else:
                info += f"Model Framework: {results['framework']}\n\n"
                info += f"Model Structure:\n{results['structure']}\n\n"
                info += f"Total Parameters: {results['param_count']}\n"
            
            self.model_info.setText(info)
            self.log("Model analysis completed.")

            # Create graph
            self.create_graph(results)

            # Create parameter table
            self.create_param_table(results)

        except Exception as e:
            self.log(f"Error updating model info: {str(e)}")
            self.log(traceback.format_exc())

    def create_graph(self, results):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        structure = results.get('structure', '')
        if isinstance(structure, str):
            lines = [line.strip() for line in structure.split('\n') if line.strip()]
            G = nx.Graph()
            layer_types = {}
            
            for i, line in enumerate(lines):
                parts = line.split(':')
                if len(parts) > 1:
                    layer_name = parts[0].strip()
                    layer_type = layer_name.split('.')[-1]  # Get the last part of the name
                    G.add_node(layer_name)
                    layer_types[layer_name] = layer_type
                
                if i > 0:
                    prev_parts = lines[i-1].split(':')
                    if len(prev_parts) > 1:
                        prev_layer_name = prev_parts[0].strip()
                        G.add_edge(prev_layer_name, layer_name)

            pos = nx.spring_layout(G, dim=3, k=0.5, iterations=50)
            
            # Color mapping
            unique_types = set(layer_types.values())
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
            color_map = dict(zip(unique_types, colors))
            
            # Draw nodes
            xs, ys, zs = [], [], []
            cs = []
            for node, (x, y, z) in pos.items():
                xs.append(x)
                ys.append(y)
                zs.append(z)
                cs.append(color_map[layer_types[node]])
            
            ax.scatter(xs, ys, zs, c=cs, s=100, edgecolors='black', alpha=0.7)
            
            # Draw edges
            for edge in G.edges():
                x = [pos[edge[0]][0], pos[edge[1]][0]]
                y = [pos[edge[0]][1], pos[edge[1]][1]]
                z = [pos[edge[0]][2]][0], pos[edge[1]][2]
                ax.plot(x, y, z, c='gray', alpha=0.2)

            ax.set_title("Model Structure")
            ax.set_axis_off()
            
            # Add a color legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=layer_type,
                            markerfacecolor=color, markersize=10)
                            for layer_type, color in color_map.items()]
            ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0), 
                    ncol=3, fontsize='x-small')

        self.canvas.draw()
        
        # Add navigation toolbar for pan, zoom, and rotate
        if not hasattr(self, 'toolbar'):
            self.toolbar = NavigationToolbar(self.canvas, self.model_tab)
            self.model_layout.addWidget(self.toolbar)

        # Set up the 3D rotation
        def rotate(angle):
            ax.view_init(azim=angle)
            self.canvas.draw()

        if not hasattr(self, 'animation'):
            self.animation = FuncAnimation(self.figure, rotate, frames=np.linspace(0, 360, 360), interval=50)
            
    def create_param_table(self, results):
        self.param_table.clear()
        self.param_table.setColumnCount(2)
        self.param_table.setHorizontalHeaderLabels(["Layer", "Shape"])

        structure = results.get('structure', '')
        if isinstance(structure, str):
            rows = []
            for line in structure.split('\n'):
                parts = line.split(':')
                if len(parts) == 2:
                    layer_name = parts[0].strip()
                    shape = parts[1].strip()
                    rows.append([layer_name, shape])
            
            self.param_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, item in enumerate(row):
                    table_item = QTableWidgetItem(str(item))
                    if j == 0:  # Layer name
                        table_item.setBackground(QColor(200, 230, 200))
                    elif j == 1:  # Shape
                        table_item.setBackground(QColor(230, 200, 200))
                    self.param_table.setItem(i, j, table_item)

        self.param_table.resizeColumnsToContents()

    def log(self, message):
        self.log_widget.append(message)
        self.log_widget.moveCursor(QTextCursor.End)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
