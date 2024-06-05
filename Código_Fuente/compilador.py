import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QToolBar, QAction, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from PyQt5.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QPainter, QTextFormat
from PyQt5.QtCore import Qt, QRegExp, QRect, QSize
import ply.lex as lex
import ply.yacc as yacc
import matplotlib.pyplot as plt
import networkx as nx

# Definición del analizador léxico con PLY
reserved = {
    'int': 'INT',
    'float': 'FLOAT',
    'boolean': 'BOOLEAN',
    'string': 'STRING',
    'void': 'VOID',
    'if': 'IF',
    'else': 'ELSE',
    'for': 'FOR',
    'return': 'RETURN',
    'true': 'TRUE',
    'false': 'FALSE',
    'begin': 'BEGIN',
    'end': 'END',
    'onoff': 'ONOFF',
    'park': 'PARK',
    'exit': 'EXIT',
    'sensor': 'SENSOR',
    'gate': 'GATE',
    'open': 'OPEN',
    'close': 'CLOSE',
    'print': 'PRINT'
}

tokens = [
    'NUMBER',
    'FLOAT_NUMBER',
    'BOOLEAN_LITERAL',
    'STRING_LITERAL',
    'IDENTIFIER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'LPAREN',
    'RPAREN',
    'SEMICOLON',
    'EQUALS',
    'LBRACE',
    'RBRACE',
    'LT',
    'GT',
    'LE',
    'GE',
    'EQ',
    'NE',
    'COMMENT'
] + list(reserved.values())

t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_SEMICOLON = r';'
t_EQUALS = r'='
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_GE = r'>='
t_EQ = r'=='
t_NE = r'!='

def t_FLOAT_NUMBER(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_BOOLEAN_LITERAL(t):
    r'true|false'
    t.value = t.value
    return t

def t_STRING_LITERAL(t):
    r'\"([^\\\n]|(\\.))*?\"'
    t.value = str(t.value)
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

def t_COMMENT(t):
    r'//.*|/\*[\s\S]*?\*/'
    pass

t_ignore = ' \t'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    t.lexer.console.appendPlainText(f"Error léxico en la línea {t.lexer.lineno}: Carácter ilegal '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

class Highlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super(Highlighter, self).__init__(parent)
        
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(Qt.darkBlue)
        keyword_format.setFontWeight(QFont.Bold)
        keywords = list(reserved.keys())
        for keyword in keywords:
            pattern = QRegExp(f"\\b{keyword}\\b")
            self.highlighting_rules.append((pattern, keyword_format))
        
        single_line_comment_format = QTextCharFormat()
        single_line_comment_format.setForeground(Qt.darkGreen)
        self.highlighting_rules.append((QRegExp("//[^\n]*"), single_line_comment_format))

        multi_line_comment_format = QTextCharFormat()
        multi_line_comment_format.setForeground(Qt.darkGreen)
        self.multi_line_comment_format = multi_line_comment_format

        self.comment_start_expression = QRegExp("/\\*")
        self.comment_end_expression = QRegExp("\\*/")

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)
        
        start_index = 0
        if self.previousBlockState() != 1:
            start_index = self.comment_start_expression.indexIn(text)
        
        while start_index >= 0:
            end_index = self.comment_end_expression.indexIn(text, start_index)
            comment_length = 0
            if end_index == -1:
                self.setCurrentBlockState(1)
                comment_length = len(text) - start_index
            else:
                comment_length = end_index - start_index + self.comment_end_expression.matchedLength()
            self.setFormat(start_index, comment_length, self.multi_line_comment_format)

symbol_table = {}

def declare_variable(identifier, var_type, lineno):
    if identifier in symbol_table:
        yacc.console.appendPlainText(f"Error semántico en la línea {lineno}: Variable '{identifier}' redeclarada")
    else:
        symbol_table[identifier] = {'type': var_type, 'initialized': False}

def check_variable(identifier, lineno):
    if identifier not in symbol_table:
        yacc.console.appendPlainText(f"Error semántico en la línea {lineno}: Variable '{identifier}' no declarada")
        return False
    return True

def check_initialized(identifier, lineno):
    if symbol_table[identifier]['initialized'] == False:
        yacc.console.appendPlainText(f"Error semántico en la línea {lineno}: Variable '{identifier}' usada sin inicializar")

def check_type_compatibility(var1, var2, lineno):
    type1 = get_type(var1)
    type2 = get_type(var2)
    if type1 is None or type2 is None:
        yacc.console.appendPlainText(f"Error semántico en la línea {lineno}: Variable '{var1}' o '{var2}' no declarada")
    elif type1 != type2:
        yacc.console.appendPlainText(f"Error semántico en la línea {lineno}: Tipos incompatibles entre '{var1}' ({type1}) y '{var2}' ({type2})")

def get_type(var):
    if isinstance(var, int):
        return 'int'
    elif isinstance(var, float):
        return 'float'
    elif isinstance(var, str):
        if var == 'true' or var == 'false':
            return 'boolean'
        elif var.startswith('"') and var.endswith('"'):
            return 'string'
        elif var in symbol_table:
            return symbol_table[var]['type']
    elif var in symbol_table:
        return symbol_table[var]['type']
    return None

def p_program(p):
    '''program : BEGIN statement_list END'''
    p[0] = ('program', p[2])

def p_statement_list(p):
    '''statement_list : statement_list statement
                      | statement'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_statement(p):
    '''statement : declaration
                 | assignment_expression SEMICOLON
                 | if_statement
                 | for_statement
                 | parking_command
                 | return_statement
                 | print_statement  
                 | SEMICOLON'''
    p[0] = p[1]

def p_declaration(p):
    '''declaration : type_specifier IDENTIFIER SEMICOLON'''
    declare_variable(p[2], p[1], p.lineno(2))
    p[0] = ('declare', p[1], p[2])

def p_type_specifier(p):
    '''type_specifier : INT
                      | FLOAT
                      | BOOLEAN
                      | STRING'''
    p[0] = p[1]

def p_assignment_expression(p):
    '''assignment_expression : IDENTIFIER EQUALS expression'''
    if check_variable(p[1], p.lineno(1)):
        check_type_compatibility(p[1], p[3], p.lineno(1))
        symbol_table[p[1]]['initialized'] = True
    p[0] = ('assign', p[1], p[3])

def p_if_statement(p):
    '''if_statement : IF LPAREN expression RPAREN compound_statement
                    | IF LPAREN expression RPAREN compound_statement ELSE compound_statement'''
    if len(p) == 6:
        p[0] = ('if', p[3], p[5])
    else:
        p[0] = ('if_else', p[3], p[5], p[7])

def p_for_statement(p):
    '''for_statement : FOR LPAREN assignment_expression_opt expression_opt SEMICOLON assignment_expression_opt RPAREN compound_statement'''
    p[0] = ('for', p[3], p[4], p[6], p[8])

def p_expression_opt(p):
    '''expression_opt : expression
                      | empty'''
    p[0] = p[1]

def p_assignment_expression_opt(p):
    '''assignment_expression_opt : assignment_expression
                                 | empty'''
    p[0] = p[1]

def p_empty(p):
    '''empty :'''
    pass

def p_parking_command(p):
    '''parking_command : GATE OPEN SEMICOLON
                       | GATE CLOSE SEMICOLON
                       | ONOFF SEMICOLON
                       | PARK SEMICOLON
                       | EXIT SEMICOLON
                       | SENSOR SEMICOLON'''
    p[0] = ('command', p[1], p[2])

def p_return_statement(p):
    '''return_statement : RETURN expression SEMICOLON'''
    p[0] = ('return', p[2])

def p_print_statement(p):
    '''print_statement : PRINT LPAREN expression RPAREN SEMICOLON'''
    p[0] = ('print', p[3])

def p_expression(p):
    '''expression : additive_expression
                  | relational_expression
                  | BOOLEAN_LITERAL
                  | STRING_LITERAL'''
    p[0] = p[1]

def p_relational_expression(p):
    '''relational_expression : additive_expression relational_op additive_expression'''
    check_type_compatibility(p[1], p[3], p.lineno(2))
    p[0] = ('relational', p[2], p[1], p[3])

def p_relational_op(p):
    '''relational_op : LT
                     | LE
                     | GT
                     | GE
                     | EQ
                     | NE'''
    p[0] = p[1]

def p_additive_expression(p):
    '''additive_expression : multiplicative_expression
                           | additive_expression PLUS multiplicative_expression
                           | additive_expression MINUS multiplicative_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        check_type_compatibility(p[1], p[3], p.lineno(2))
        p[0] = ('binary-op', p[2], p[1], p[3])

def p_multiplicative_expression(p):
    '''multiplicative_expression : primary_expression
                                 | multiplicative_expression TIMES primary_expression
                                 | multiplicative_expression DIVIDE primary_expression'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        check_type_compatibility(p[1], p[3], p.lineno(2))
        p[0] = ('binary-op', p[2], p[1], p[3])

def p_primary_expression(p):
    '''primary_expression : IDENTIFIER
                          | NUMBER
                          | FLOAT_NUMBER
                          | BOOLEAN_LITERAL
                          | STRING_LITERAL
                          | LPAREN expression RPAREN'''
    if len(p) == 2:
        if isinstance(p[1], str) and p[1] in symbol_table:
            check_initialized(p[1], p.lineno(1))
        p[0] = p[1]
    else:
        p[0] = p[2]

def p_compound_statement(p):
    '''compound_statement : LBRACE statement_list RBRACE'''
    p[0] = ('compound', p[2])

def p_error(p):
    if p:
        yacc.console.appendPlainText(f"Error sintáctico en la línea {p.lineno}: Token inesperado '{p.value}'")
    else:
        yacc.console.appendPlainText("Error sintáctico en el final del archivo")

parser = yacc.yacc()

def generar_grafo(tokens):
    G = nx.DiGraph()
    stack = []
    for token in tokens:
        if token.type in ['BEGIN', 'IF', 'ELSE', 'FOR', 'LBRACE']:
            stack.append(token)
            G.add_node(token.value, label=token.value)
            if len(stack) > 1:
                G.add_edge(stack[-2].value, token.value)
        elif token.type in ['END', 'RBRACE']:
            stack.pop()
        else:
            G.add_node(token.value, label=token.value)
            if stack:
                G.add_edge(stack[-1].value, token.value)

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    plt.show()

class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event):
        self.editor.lineNumberAreaPaintEvent(event)

class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineNumberArea = LineNumberArea(self)

        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.cursorPositionChanged.connect(self.highlightCurrentLine)

        self.updateLineNumberAreaWidth(0)

    def lineNumberAreaWidth(self):
        digits = 1
        max = self.blockCount()
        while max >= 10:
            max //= 10
            digits += 1
        space = 3 + self.fontMetrics().width('9') * digits
        return space

    def updateLineNumberAreaWidth(self, _):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect, dy):
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(0, rect.y(), self.lineNumberArea.width(), rect.height())

        if rect.contains(self.viewport().rect()):
            self.updateLineNumberAreaWidth(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height()))

    def lineNumberAreaPaintEvent(self, event):
        painter = QPainter(self.lineNumberArea)
        painter.fillRect(event.rect(), Qt.lightGray)

        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.setPen(Qt.black)
                painter.drawText(0, int(top), self.lineNumberArea.width(), int(self.fontMetrics().height()),
                                 Qt.AlignRight, number)

            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            blockNumber += 1

    def highlightCurrentLine(self):
        extraSelections = []

        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()

            lineColor = QColor(Qt.yellow).lighter(160)

            selection.format.setBackground(lineColor)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extraSelections.append(selection)

        self.setExtraSelections(extraSelections)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Compilador de Estacionamiento Automático")
        
        self.editor = CodeEditor()
        self.editor.setFont(QFont("Courier", 12))

        self.console = QPlainTextEdit(self)
        self.console.setFont(QFont("Courier", 10))
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: black; color: white;")

        self.create_toolbar()

        layout = QVBoxLayout()
        layout.addWidget(self.editor)
        layout.addWidget(self.console)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.highlighter = Highlighter(self.editor.document())

    def create_toolbar(self):
        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)

        load_action = QAction("Cargar", self)
        load_action.triggered.connect(self.load_file)
        toolbar.addAction(load_action)

        analyze_action = QAction("Analizar", self)
        analyze_action.triggered.connect(self.analyze_code)
        toolbar.addAction(analyze_action)

        graph_action = QAction("Mostrar Grafo", self)
        graph_action.triggered.connect(self.show_graph)
        toolbar.addAction(graph_action)

        clear_action = QAction("Limpiar", self)
        clear_action.triggered.connect(self.clear_editor)
        toolbar.addAction(clear_action)

        save_action = QAction("Guardar", self)
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)

    def zoom_in(self):
        self.editor.zoomIn()

    def zoom_out(self):
        self.editor.zoomOut()

    def load_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Cargar Archivo", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                self.editor.setPlainText(file.read())

    def analyze_code(self):
        self.console.clear()
        global symbol_table
        symbol_table = {}
        code = self.editor.toPlainText()
        lexer.input(code)
        lexer.console = self.console
        yacc.console = self.console
        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append(tok)
        if not tokens:
            self.console.appendPlainText("No se encontraron tokens válidos.")
        try:
            parser.parse(code)
            self.console.appendPlainText("Análisis léxico, sintáctico y semántico completado.")
        except Exception as e:
            self.console.appendPlainText(f"Error durante el análisis sintáctico: {str(e)}")

    def show_graph(self):
        code = self.editor.toPlainText()
        lexer.input(code)
        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append(tok)
        if tokens:
            generar_grafo(tokens)
        else:
            self.console.appendPlainText("No se encontraron tokens válidos.")

    def clear_editor(self):
        self.editor.clear()
        self.console.clear()

    def save_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Guardar Archivo", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            with open(file_name, 'w') as file:
                file.write(self.editor.toPlainText())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
