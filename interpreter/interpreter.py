class AST:
    """
    Abstract Syntax Tree manager
    Tree - list representation of AST
    Labels - dict of goto labels
    """

    def __init__(self, filename):
        pass
        self.tree = []
        self.labels = {}
        self.construct_ast(filename)

    def construct_ast(self, filename):
        """
        Takes flattened AST as input.
        Outputs AST as list, and labels as dict {label : line_number}
        """
        with open(filename) as f:
            line_count = 0
            for line in f:
                line = line.strip().split()
                if line[0] == "Label":
                    self.labels[line[1]] = line_count - 1
                self.tree.append(line)
                line_count += 1


class Interpreter:
    """
    Interprets code with a certain given grammar.
    Vars - local variables
    Tree, Labels - fields from AST
    PC - program counter (current line)
    Operations - dict of possible operations
    """

    def __init__(self):
        self.vars = {}
        self.tree = []
        self.labels = {}
        self.pc = 0
        self.operations = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x / y,
            "<": lambda x, y: x < y,
            ">": lambda x, y: x > y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y
        }

    def start(self, filename):
        """
        Read input AST file line by line.
        Return result of the program execution
        """
        ast = AST(filename)
        self.tree = ast.tree
        self.labels = ast.labels

        # flag to stop and return current value
        stop = False
        while not stop:
            # interpret line starting from line 0
            stop, value = self.interpret(self.pc)

        return value

    def interpret(self, line):
        """
        Interprets one line from flattened AST with the given line_number
        Returns stop flag and result of the execution
        """

        # change program counter to current line
        self.pc = line
        # last line condition
        if line >= len(self.tree):
            return None

        # current token name
        name = self.tree[line][0]

        if name in ["Program", "Jump", "Semicolon"]:
            self.pc += 1  # move to the next line
            return False, None
        elif name == "BasicBlock":
            self.pc += 2  # move 2 lines ahead, skipping the label tag
            return False, None
        elif name == "Assignment":
            _, var_name = self.interpret(line + 1)  # get left variable
            _, expression = self.interpret(line + 2)  # get right expression
            self.vars[var_name] = expression  # update the variable
            self.pc += 1
            return False, None
        elif name == "Expression":
            # check if the expression is constant value or variable
            if self.tree[line + 1][0] == "Const":
                const = int(self.interpret(line + 1)[1])  # interpret as const
            else:
                var_name = self.interpret(line + 1)[1]  # interpret as variable
                const = int(self.vars[var_name])

            # check if the expresssion has any operators
            if self.tree[line + 2][0] == "Op":
                operation = self.interpret(line + 2)[1]
                if self.tree[line + 3][0] in ["Const", "Expression"]:
                    expr = int(self.interpret(line + 3)[1])
                else:
                    var_name = self.interpret(line + 3)[1]
                    expr = int(self.vars[var_name])
                return False, operation(const, expr)
            else:
                return False, const
        elif name == "GoTo":
            line_number = self.labels[self.tree[line][1]]
            self.pc = line_number  # change program counter to move to desired line
            return False, None
        elif name == "If":
            _, expr = self.interpret(line + 1)
            if expr:
                self.pc += 1  # if the expression is true
            else:
                self.pc += 2  # if the expression is false
            return False, None
        elif name == "Return":
            return True, self.interpret(line + 1)[1]  # return stop flag and value of expression
        elif name == "Constant":
            return False, self.tree[line][1]
        elif name == "Var":
            var_name = self.tree[line][1]
            return False, var_name
        elif name == "EmptyVar":
            var_name = self.tree[line][1]  # enter the value for the empty variable
            self.vars[var_name] = input()
            self.pc += 1
            return False, self.interpret(line + 1)
        elif name == "Op":
            return False, self.operations[self.tree[line][1]]  # get the operation from the dict


# Check the interpreter with the euclidean algorithm from euc.txt
it = Interpreter()
val = it.start("euc.txt")
print(val)
