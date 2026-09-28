class VistaBackground {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'vista-background-canvas';
        document.body.insertBefore(this.canvas, document.body.firstChild);
        this.ctx = this.canvas.getContext('2d');
        
        this.particles = [];
        this.numParticles = 50; // Low particle count for performance
        this.connectionDistance = 150;
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        this.initParticles();
        
        this.isRunning = true;
        
        // Performance optimization: Pause when not visible
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                this.isRunning = true;
                this.animate();
            } else {
                this.isRunning = false;
            }
        });
        
        // Limit FPS to ~30 for lower CPU usage
        this.fpsInterval = 1000 / 30;
        this.then = Date.now();
        
        this.animate();
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    initParticles() {
        this.particles = [];
        for (let i = 0; i < this.numParticles; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5, // Slow movement
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 1.5 + 0.5
            });
        }
    }
    
    updateParticles() {
        for (let p of this.particles) {
            p.x += p.vx;
            p.y += p.vy;
            
            // Wrap around edges
            if (p.x < 0) p.x = this.canvas.width;
            if (p.x > this.canvas.width) p.x = 0;
            if (p.y < 0) p.y = this.canvas.height;
            if (p.y > this.canvas.height) p.y = 0;
        }
    }
    
    drawParticles() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Background color handled by CSS
        
        this.ctx.fillStyle = 'rgba(15, 118, 110, 0.5)'; // VISTA Teal
        
        for (let i = 0; i < this.numParticles; i++) {
            let p1 = this.particles[i];
            this.ctx.beginPath();
            this.ctx.arc(p1.x, p1.y, p1.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            for (let j = i + 1; j < this.numParticles; j++) {
                let p2 = this.particles[j];
                let dx = p1.x - p2.x;
                let dy = p1.y - p2.y;
                let dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < this.connectionDistance) {
                    this.ctx.beginPath();
                    this.ctx.strokeStyle = `rgba(15, 118, 110, ${0.15 * (1 - dist / this.connectionDistance)})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.moveTo(p1.x, p1.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.stroke();
                }
            }
        }
    }
    
    animate() {
        if (!this.isRunning) return;
        
        requestAnimationFrame(() => this.animate());
        
        let now = Date.now();
        let elapsed = now - this.then;
        
        if (elapsed > this.fpsInterval) {
            this.then = now - (elapsed % this.fpsInterval);
            this.updateParticles();
            this.drawParticles();
        }
    }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
    new VistaBackground();
});
