
                            function toggleCamType() {
                                const type = document.getElementById('cam-type').value;
                                const srcLabel = document.getElementById('cam-source-label');
                                const srcInput = document.getElementById('cam-source');
                                const hint = document.getElementById('cam-source-hint');
                                if (type === 'ip') {
                                    srcLabel.innerText = 'RTSP / HTTP URL';
                                    srcInput.placeholder = 'rtsp://admin:pass@192.168.1.100:554/stream1';
                                    hint.classList.add('hidden');
                                } else {
                                    srcLabel.innerText = 'USB Device Index';
                                    srcInput.placeholder = '0  (first wired camera)   or   1, 2, 3…';
                                    hint.classList.remove('hidden');
                                }
                            }
                        