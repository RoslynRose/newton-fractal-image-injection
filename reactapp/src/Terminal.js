class Terminal {
    constructor(elementId) {
        this.container = document.getElementById(elementId);
        this.inputLine = null;
        this.prompt = '>';
        this.commands = {};
    }

    initialize() {
        this.container.innerHTML = '<div class="output"></div>';
        this.inputLine = this.createInputLine();
        this.container.appendChild(this.inputLine);
        this.focusInput();

        this.container.addEventListener('click', (e) => {
            if (e.target !== this.inputLine.querySelector('.input')) {
                this.focusInput();
            }
        });


        this.addCommand(
            'help', 
            (arg) => arg ? this.commandHelp(arg) : `Available commands: ${Object.keys(this.commands).join(', ')}`, 
            'Displays information about available commands or details about a specific command.'
        );
        this.addCommand(
            'clear',
            () => this.clearTerminal(),
            'Clears the terminal output.'
        );
    }

    clearTerminal() {
        const outputDiv = this.container.querySelector('.output');
        outputDiv.innerHTML = '';
        return '';
    }

    createInputLine() {
        const line = document.createElement('div');
        line.className = 'input-line';
        const span = document.createElement('span');
        span.innerHTML = this.prompt;
        const inputDiv = document.createElement('div');
        inputDiv.className = 'input';
        inputDiv.setAttribute('contenteditable', 'true');
        inputDiv.addEventListener('keydown', (e) => this.handleInput(e));

        line.appendChild(span);
        line.appendChild(inputDiv);
        return line;
    }

    handleInput(event) {
        if (event.key === 'Enter') {
            event.preventDefault();  // Prevent new lines.
            const input = event.target.innerText;
            this.runCommand(input);
            event.target.innerText = '';
        }
    }

    appendOutput(output) {
        const outputLine = document.createElement('div');
        outputLine.className = 'output-line';
        outputLine.innerText = output;
        this.container.querySelector('.output').appendChild(outputLine);
        this.scrollToBottom();  // Ensure the terminal scrolls to the new output.
        this.focusInput();
    }

    focusInput() {
        this.inputLine.querySelector('.input').focus();
    }

    addCommand(name, func, help) {
        if (typeof func === 'function') {
            this.commands[name] = { func, help };
            this.commands['help'].func = (arg) => arg ? this.commandHelp(arg) : `Available commands: ${Object.keys(this.commands).join(', ')}`;
        } else {
            console.error('Provided command logic is not a function');
        }
        console.log("Commands after adding: ", this.commands);
    }

    runCommand(input) {
        console.log("Commands at the time of execution: ", this.commands);
        this.appendOutput(`${this.prompt} ${input}`);

        // Use RegEx to split the input, keeping quoted strings intact
        const args = input.match(/(".*?"|\S)+/g) || [];
        const command = args.shift();

        // Remove quotes from quoted arguments
        const parsedArgs = args.map(arg => {
            if (arg.startsWith('"') && arg.endsWith('"')) {
                return arg.slice(1, -1);
            }
        return isNaN(arg) ? arg : Number(arg);
        });

        if (command === 'help') {
            console.log('Entered help command block');
            if (args.length > 0) {
                const cleanedArg = args[0].trim();  // Add this line to clean up the input
                console.log(`Argument provided: '${cleanedArg}'`); // Use cleanedArg here
                console.log(`Type of argument: ${typeof cleanedArg}`); // And here
                console.log(`Available commands: ${Object.keys(this.commands)}`);
                console.log(`Does the command exist? ${this.commands.hasOwnProperty(cleanedArg)}`); // And here
                if (this.commands.hasOwnProperty(cleanedArg)) {
                    this.appendOutput(this.commands[cleanedArg].help || `No help available for ${cleanedArg}`); // And here
                } else {
                    this.appendOutput(`Unknown command: ${cleanedArg}`); // And finally here
                }
                return;
            } else {
                this.appendOutput(`Available commands: ${Object.keys(this.commands).join(', ')}`);
                return;
            }
        }


        if (this.commands.hasOwnProperty(command)) {
            const output = this.commands[command].func(...parsedArgs);
            this.appendOutput(output);
        } else {
            this.appendOutput(`Unknown command: ${command}`);
        }
    }
    commandHelp(command) {
        if (this.commands.hasOwnProperty(command)) {
            return this.commands[command].help || `No help available for ${command}`;
        }
        return `Unknown command: ${command}`;
    }

    removeCommand(name) {
        delete this.commands[name];
    }
    scrollToBottom() {
        this.container.scrollTop = this.container.scrollHeight;
    }
}

export default Terminal;

